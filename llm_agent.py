import subprocess
import os
import re
import sys
from datetime import datetime  # moved to top for cleanliness
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class AutoResearchLLMAgent:
    def __init__(self, max_iterations=5, improvement_threshold=0.001):
        self.client = OpenAI()
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.best_metric = 1.6833
    # ---------- FILE & SCRIPT HELPERS ----------
    def run_script(self, script_name):
        print(f"Executing {script_name}...")
        result = subprocess.run(
            [sys.executable, script_name], capture_output=True, text=True
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
        # By using HEAD instead of HEAD~1, it only throws away UNCOMMITTED bad code
        subprocess.run(["git", "reset", "--hard", "HEAD"], capture_output=True)

    # ---------- PROGRAM.MD UPDATE ----------
    def update_program_md(self, model_name, features, sharpe, notes):
        md_path = "program.md"
        content = self.read_file(md_path)
        header_line = "| Date | Model | Features | Acc (Test) | Strat Return | Market Return | Sharpe | Notes |"
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

        from datetime import datetime
        today_date = datetime.today().strftime('%Y-%m-%d')
        new_row = f"| {today_date} | {model_name} | {features} | - | - | - | {sharpe:.3f} | {notes} |"
        lines.insert(insert_idx + 1, new_row)
        self.write_file(md_path, "\n".join(lines))
        print("Updated program.md with new experiment row.")

    # ---------- LLM INTERACTION ----------
    def prompt_hypothesis(self, program_md, train_code, backtest_code):
        prompt = f"""You are an AI researcher improving a financial ML pipeline.
Current project log (program.md):
{program_md}

Current training script (train.py):
```python
{train_code}
# your code here
Current backtest script (backtest.py):

Python
{backtest_code}
Based on the logs and current code, propose a SINGLE architectural improvement to train.py that will increase the Sharpe Ratio.
Output the FULL, updated train.py python code inside a python code block.

CRITICAL RULES:
1. DO NOT rename the `FinanceModel` class. It MUST stay exactly `FinanceModel`.
2. Do not change the data loading paths (keep them exactly as 'data/train.csv').
3. Do not forget to define the `features` array.
4. Output ONLY valid python code.
5. PREVENT LAZY PREDICTIONS: The model previously predicted 1 every day (Sharpe 1.68), but your recent fixes caused it to overcorrect and predict 0 every day (Sharpe 0.00). You must balance the model so it outputs a healthy mix of 1s and 0s. Try adjusting class weights carefully, normalizing the input features (BatchNorm/StandardScaler), or tuning the learning rate.6. IF USING LSTM/GRU: You MUST reshape the 2D input `x` into a 3D tensor `(batch_size, 1, features)` inside the `forward` pass before feeding it to the RNN.
7. PYTORCH LOSS RULES: `nn.BCELoss` does NOT accept `pos_weight`. If you use `pos_weight`, you MUST switch to `nn.BCEWithLogitsLoss` and remove the final `Sigmoid` layer.
"""

        print("Querying LLM for hypothesis...")
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content

    # ---------- CORE EXPERIMENT LOOP ----------
    def run(self):
        print("Starting AutoResearch Agent Loop...")
        self.git_pre_experiment_commit()  # baseline commit

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")

            # 1. Read context
            program_md = self.read_file("program.md")
            train_code = self.read_file("train.py")
            backtest_code = self.read_file("backtest.py")

            # 2. Form hypothesis and get new code
            llm_response = self.prompt_hypothesis(program_md, train_code, backtest_code)
            new_train_code = self.extract_code_block(llm_response)

            # 3. Write new code to train.py
            self.write_file("train.py", new_train_code)
            print("Proposed changes written to train.py")

            # 4. Run experiment
            train_success, _, train_err = self.run_script("train.py")
            if not train_success:
                print("Training failed with an error. Reverting...")
                print(f"Error snippet: {train_err[-300:]}")
                self.git_revert()
                continue

            backtest_success, bt_out, bt_err = self.run_script("backtest.py")
            if not backtest_success:
                print("Backtesting failed with an error. Reverting...")
                print(f"Error snippet: {bt_err[-300:]}")
                self.git_revert()
                continue

            # 5. Evaluate results
            new_metric = self.parse_metric(bt_out)
            if new_metric is None:
                print("Could not find BACKTEST_METRIC in output. Reverting...")
                self.git_revert()
                continue

            print(f"New Sharpe: {new_metric:.4f} | Best Sharpe: {self.best_metric:.4f}")

            # 6. Keep or Revert
            if new_metric > self.best_metric + self.improvement_threshold:
                print("Improvement found! Committing new baseline...")
                self.best_metric = new_metric
                # --- SWAP THESE TWO LINES ---
                self.update_program_md("LLM Proposed Model", "Auto-updated", new_metric, "Iteration success")
                self.git_keep(f"Sharpe improved to {new_metric:.4f}")
                # ----------------------------
            else:
                print("No significant improvement. Reverting to baseline...")
                self.git_revert()


if __name__ == "__main__":
    agent = AutoResearchLLMAgent()
    agent.run()