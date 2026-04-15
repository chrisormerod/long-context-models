import os
import subprocess
import sys
import time


def main():
    for run_id in range(10):
        output_csv = f"mbert_trunc_run_{run_id}.csv"

        cmd = [
            sys.executable,
            "run_one_training.py",
            "--run-id",
            str(run_id),
            "--output-csv",
            output_csv,
        ]

        print(f"Starting run {run_id}...")
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Run {run_id} failed with exit code {result.returncode}")

        # Small pause can help the system fully release resources
        time.sleep(2)
        print(f"Finished run {run_id}, wrote {output_csv}")


if __name__ == "__main__":
    main()