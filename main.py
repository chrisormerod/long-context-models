import os
import subprocess
import sys
import time

sequence = [
            {"script":"train_mamba_trunc","csv":"outputs/mamba_trunc"},
            {"script":"train_mbert_trunc","csv":"outputs/mbert_trunc"}
            ]


def main():
    """
    Execute a sequence of model training runs with varying maximum sequence lengths.
    
    This script orchestrates the training of multiple models (Mamba and ModernBERT)
    across a range of maximum sequence lengths and multiple runs per configuration.
    It checks for existing output files to avoid redundant computation.
    
    Configuration:
        - Models: Mamba truncated and ModernBERT truncated variants
        - Max lengths: 64 to 2048 tokens (in steps of 64)
        - Runs per configuration: 10
    
    Output:
        - CSV files with predictions saved to specified output directories
        - Execution time and status logged to stdout
    
    Raises:
        RuntimeError: If any subprocess exits with a non-zero return code
    """
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
