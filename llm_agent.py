import subprocess
import os
import re
import sys
from openai import OpenAI


class AutoResearchLLMAgent:
    def __init__(self, max_iterations=5, improvement_threshold=0.001):
        self.client = OpenAI()
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.best_metric = -float("inf")

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
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)

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