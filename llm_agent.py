import subprocess
import os
import re
from openai import OpenAI

class AutoResearchLLMAgent:
    def __init__(self, max_iterations=5):
        # Initialize the LLM client (Ensure OPENAI_API_KEY is in your environment variables)
        self.client = OpenAI()
        self.max_iterations = max_iterations
        self.history = []

    def run_script(self, script_name):
        """Runs a python script and returns the output."""
        print(f"Executing {script_name}...")
        result = subprocess.run(["python", script_name], capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr

    def read_file(self, filepath):
        with open(filepath, 'r') as f:
            return f.read()

    def write_file(self, filepath, content):
        with open(filepath, 'w') as f:
            f.write(content)

    def extract_code_block(self, text):
        """Extracts python code from LLM's markdown response."""
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else text

    def prompt_llm(self, current_code, backtest_output):
        """Asks the LLM to improve the model based on the latest results."""
        print("\n[Brain] Asking LLM for a better architecture...")
        
        system_prompt = (
            "You are an expert quantitative researcher and AI engineer. "
            "Your goal is to modify a PyTorch training script to improve the Sharpe ratio "
            "of an S&P 500 directional trading strategy. "
            "You are only allowed to output the raw python code for train.py wrapped in ```python blocks. "
            "Do NOT change the feature inputs, only change the PyTorch model architecture, loss function, or optimizer."
        )

        user_prompt = f"""
        Here is the current train.py:
        ```python
        {current_code}
        ```
        
        Here were the results of running backtest.py on this model:
        {backtest_output}
        
        Please rewrite train.py. Improve the neural network architecture (e.g., try deeper networks, dropouts, different learning rates, or advanced architectures) to increase the Strategy Return and Sharpe Ratio. Output the full, complete new train.py code.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o", # Or gpt-4-turbo, etc.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def run_loop(self):
        print("--- Starting LLM Autoresearch Loop ---")
        
        # Step 1: Ensure data is prepared initially
        self.run_script("prepare.py")

        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            
            # Step 2: Train the current model
            success, train_out, train_err = self.run_script("train.py")
            if not success:
                print(f"Training failed. Error:\n{train_err}")
                break
                
            # Step 3: Backtest the model
            success, backtest_out, backtest_err = self.run_script("backtest.py")
            if not success:
                print(f"Backtesting failed. Error:\n{backtest_err}")
                break
                
            print(backtest_out)
            
            # Step 4: LLM Generation (The "AI" part)
            current_train_code = self.read_file("train.py")
            llm_response = self.prompt_llm(current_train_code, backtest_out)
            
            # Step 5: Extract and apply new code
            new_train_code = self.extract_code_block(llm_response)
            self.write_file("train.py", new_train_code)
            
            print("[Brain] Successfully wrote new hypothesis to train.py.")

        print("\n--- Research Loop Complete ---")

if __name__ == "__main__":
    agent = AutoResearchLLMAgent(max_iterations=3)
    agent.run_loop()