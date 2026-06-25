import os
import subprocess
import sys
import time

sequence = [
            {"script":"train_mamba_trunc","csv":"outputs/mamba_trunc"},
            {"script":"train_mbert_trunc","csv":"outputs/mbert_trunc"}
            ]


def main():
    for config in sequence:
        
        script = "scripts."+config['script']
        csv = config['csv']
        for ml in [64*x for x in range(1,33)]:
            for run_id in range(10):
                output_csv = csv+f"_{ml}_{run_id}.csv"
                if os.path.exists(output_csv):
                    print(f"Run {output_csv} exists")
                else:
                    cmd = [
                        sys.executable,
                        "-m",
                        script,
                        "--run-id",
                        str(run_id),
                        "--output-csv",
                        output_csv,
                        "--max-length",
                        str(ml)
                    ]
                    
                    print(f"Starting {script} run {run_id}...")
                    result = subprocess.run(cmd, check=False)
            
                    if result.returncode != 0:
                        raise RuntimeError(f"Run {run_id} failed with exit code {result.returncode}")
            
                    # Small pause can help the system fully release resources
                    time.sleep(2)
                    print(f"Finished run {run_id}, wrote {output_csv}")


if __name__ == "__main__":
    main()