import os
import re
import sys
import subprocess
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class AutoResearchLLMAgent:
    def __init__(self, max_iterations=15, improvement_threshold=0.001):
        self.client = OpenAI()
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.best_metric = 0

    # ---------- FILE & SCRIPT HELPERS ----------
    def run_script(self, script_name, args=None):
        if args is None:
            args = []
            
        print(f"Executing {script_name}...")

        # Append the additional arguments to the command list
        command = [sys.executable, script_name] + args

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        return result.returncode == 0, result.stdout, result.stderr

    def read_file(self, filepath):
        if not os.path.exists(filepath):
            return ""

        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, filepath, content):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def extract_code_block(self, text):
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)

        return match.group(1) if match else text

    def extract_summary(self, text):
        """Extracts the hyperparameter summary provided by the LLM."""

        pattern = r"<summary>(.*?)</summary>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        # Clean up newlines so it fits nicely in the Markdown table
        if match:
            return match.group(1).strip().replace("\n", " ")

        return "Auto-updated architecture and hyperparameters."

    # ---------- METRIC PARSING ----------
    def parse_metric(self, stdout):
        match = re.search(r"BACKTEST_METRIC: sharpe=([\d\.\-]+)", stdout)

        if match:
            return float(match.group(1))

        return None

    # ---------- GIT HELPERS ----------
    def git_pre_experiment_commit(self):
        subprocess.run(["git", "add", "."], capture_output=True)

        subprocess.run(
            ["git", "commit", "-m", "pre-experiment baseline"],
            capture_output=True,
        )

    def git_keep(self, description):
        subprocess.run(["git", "add", "."], capture_output=True)

        subprocess.run(
            ["git", "commit", "--amend", "-m", f"Experiment: {description}"],
            capture_output=True,
        )

    def git_revert(self):
        subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True)

    # ---------- PROGRAM.MD UPDATE ----------
    def update_program_md(self, model_name, features, sharpe, notes):
        md_path = "program.md"
        content = self.read_file(md_path)

        header_line = (
            "| Date | Model | Features | Acc (Test) | Strat Return | "
            "Market Return | Sharpe | Notes |"
        )

        if header_line not in content:
            print("[Warning] Experiment table header not found.")
            return

        lines = content.split("\n")
        insert_idx = -1
        header_idx = -1

        for i, line in enumerate(lines):
            if line.strip() == header_line.strip():
                header_idx = i

            if header_idx != -1 and i > header_idx + 1 and line.startswith("|"):
                if "---" not in line and "Date" not in line:
                    insert_idx = i

        if insert_idx == -1:
            insert_idx = header_idx + 2

        today_date = datetime.today().strftime("%Y-%m-%d")

        new_row = (
            f"| {today_date} | {model_name} | {features} | - | - | - | "
            f"{sharpe:.3f} | {notes} |"
        )

        lines.insert(insert_idx + 1, new_row)

        self.write_file(md_path, "\n".join(lines))

        print(f"Updated program.md with new experiment row. Notes: {notes}")

    # ---------- LLM INTERACTION & VALIDATION ----------
    def get_hypothesis_prompt(self, program_md, train_code, backtest_code):
        return f"""You are an AI researcher improving a financial ML pipeline.

Current project log (program.md):
{program_md}

Current training script (train.py):
```python
{train_code}
```

Current backtest script (backtest.py):
```python
{backtest_code}
```

Based on the logs and current code, propose an architectural or hyperparameter improvement to train.py that will increase the Sharpe Ratio.

Output the FULL, updated train.py python code inside a python code block.

CRITICAL RULES:

- DO NOT rename the FinanceModel class. It MUST stay exactly FinanceModel.
- Do not change the data loading paths (keep them exactly as 'data/train.csv').
- Do not forget to define the features array.
- Output ONLY valid python code.

PREVENT LAZY PREDICTIONS:
The model previously predicted 1 every day (Sharpe 1.68), but your recent fixes caused it to overcorrect and predict 0 every day (Sharpe 0.00).
You must balance the model so it outputs a healthy mix of 1s and 0s.

IF USING LSTM/GRU:
You MUST reshape the 2D input x into a 3D tensor (batch_size, 1, features) inside the forward pass before feeding it to the RNN.

PYTORCH LOSS RULES:
- nn.BCELoss does NOT accept pos_weight.
- If you use pos_weight, you MUST switch to nn.BCEWithLogitsLoss and remove the final Sigmoid layer.

HYPERPARAMETER TUNING:
Actively search for better hyperparameters.
Do not leave lr=0.001, epochs=500, or dropout rates hardcoded to their previous values.
Adjust them in your generated code to explore the search space and find better local minima.

EXPERIMENT TRACKING:
Before your Python code block, you MUST provide a brief 1-2 sentence summary of your changes wrapped in <summary> tags.
This summary MUST explicitly state the hyperparameters you chose.
Example:
<summary>Switched to GRU, changed lr to 0.005, increased epochs to 600, and set dropout to 0.3.</summary>

This will be logged to program.md so you know what you have already tried.
"""

    def generate_and_validate_code(self, base_prompt, max_retries=3):
        """
        Asks the LLM for new code, runs it,
        and feeds back any errors for debugging.

        Returns:
            tuple: (success_boolean, final_stdout, final_stderr, summary_string)
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI quantitative researcher and "
                    "PyTorch developer. Always output complete, runnable "
                    "Python code wrapped in python code blocks."
                ),
            },
            {
                "role": "user",
                "content": base_prompt,
            },
        ]

        for attempt in range(max_retries):
            print(f"\n--- Code Generation Attempt {attempt + 1}/{max_retries} ---")

            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                temperature=0.7,
            )

            llm_reply = response.choices[0].message.content

            # Extract summary for hyperparameter tracking
            summary = self.extract_summary(llm_reply)

            # Extract and save code
            new_train_code = self.extract_code_block(llm_reply)
            self.write_file("train.py", new_train_code)

            print("Proposed changes written to train.py. Validating syntax and runtime...")

            success, stdout, stderr = self.run_script("train.py")

            if success:
                print(" train.py executed successfully!")
                return True, stdout, stderr, summary

            print(" train.py failed. Extracting traceback and requesting fix from LLM...")
            print(f"Error snippet: {stderr[-300:]}")

            error_message = (
                "Your previous code execution failed with the following traceback:\n"
                f"```text\n{stderr}\n```\n"
                "Please analyze the error carefully, fix the bug(s), and provide "
                "the fully corrected, complete train.py code. "
                "Make sure to include your <summary> tag again."
            )

            messages.append({"role": "assistant", "content": llm_reply})
            messages.append({"role": "user", "content": error_message})

        print(f" Max retries ({max_retries}) reached. The LLM could not fix the code.")

        return False, "", stderr, ""

    # ---------- CORE EXPERIMENT LOOP ----------
    def run(self):
        print("Starting AutoResearch Agent Loop...")
        self.git_pre_experiment_commit()

        for i in range(self.max_iterations):
            print(f"\n========== Iteration {i+1}/{self.max_iterations} ==========")

            program_md = self.read_file("program.md")
            train_code = self.read_file("train.py")
            backtest_code = self.read_file("backtest.py")

            base_prompt = self.get_hypothesis_prompt(program_md, train_code, backtest_code)

            print("Querying LLM for hypothesis and validating...")
            train_success, train_out, train_err, exp_summary = self.generate_and_validate_code(base_prompt, max_retries=3)
            
            if not train_success:
                print("Training validation failed after max retries. Reverting to baseline...")
                self.git_revert()
                continue

            # PHASE 1: Run Backtest on VALIDATION data during the iterative loop
            print("Running Backtest on Validation Data...")
            backtest_success, bt_out, bt_err = self.run_script("backtest.py", ["--val"])
            if not backtest_success:
                print("Validation Backtesting failed with an error. Reverting...")
                print(f"Error snippet: {bt_err[-300:]}")
                self.git_revert()
                continue

            new_metric = self.parse_metric(bt_out)
            if new_metric is None:
                print("Could not find BACKTEST_METRIC in output. Reverting...")
                self.git_revert()
                continue

            print(f"New Validation Sharpe: {new_metric:.4f} | Best Val Sharpe: {self.best_metric:.4f}")

            if new_metric > self.best_metric + self.improvement_threshold:
                print("✨ Improvement found! Committing new baseline...")
                self.best_metric = new_metric
                
                self.update_program_md("LLM Proposed Model", "Auto-updated", new_metric, exp_summary)
                self.git_keep(f"Val Sharpe improved to {new_metric:.4f}")
            else:
                print("No significant improvement. Reverting to baseline...")
                self.git_revert()

        # PHASE 2: Lock-in the best model and evaluate on unseen TEST data
        print("\n========== RESEARCH PHASE COMPLETE ==========")
        print("Running final evaluation on LOCKED TEST SET...")
        test_success, test_out, test_err = self.run_script("backtest.py", ["--test"])
        
        if test_success:
            final_sharpe = self.parse_metric(test_out)
            print("==================================================")
            print(f" FINAL OUT-OF-SAMPLE TEST SHARPE RATIO: {final_sharpe:.4f}")
            print("==================================================")
            
            # Optionally, append this final result to program.md
            self.update_program_md("FINAL TEST RESULT", "Locked-in Model", final_sharpe, "Out-of-sample performance")
        else:
            print("Final test backtest failed.")
            print(f"Error snippet: {test_err[-300:]}")


if __name__ == "__main__":
    agent = AutoResearchLLMAgent()
    agent.run()
