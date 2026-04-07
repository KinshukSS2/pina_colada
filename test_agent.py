from __future__ import annotations

import os
import subprocess


def run_matrix() -> None:
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    tasks = ["easy", "medium", "hard", "chaos"]

    print("========================================")
    print(" Starting Diagnostic Evaluation Matrix  ")
    print("========================================")

    for model in models:
        for task in tasks:
            print(f"\n[EVALUATING] Model: {model} | Task: {task}")
            os.environ["MODEL_NAME"] = model
            try:
                subprocess.run(["python", "inference.py", task], check=True)
            except subprocess.CalledProcessError:
                print(f"[FAIL] Run crashed for {model} on {task}.")


if __name__ == "__main__":
    run_matrix()
