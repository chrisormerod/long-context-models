import os
import subprocess
import sys
import time

sequence = [{"script":"train_mbert","csv":"outputs/mbert"},
            {"script":"train_mbert_trunc","csv":"outputs/mbert_trunc"},
            {"script":"train_mamba","csv":"outputs/mamba"},
            {"script":"train_mamba_trunc","csv":"outputs/mamba_trunc"},
            {"script":"train_mamba","csv":"outputs/mamba"},
            {"script":"train_mamba_trunc","csv":"outputs/mamba_trunc"}]


def main():
    for 
        for run_id in range(10):
            output_csv = f"mamba_run_{run_id}.csv"
    
            cmd = [
                sys.executable,
                "-m",
                "scripts.run_one_training",
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