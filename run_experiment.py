import os
import yaml
import itertools
import subprocess
import argparse
import time

def run_experiment(config_path, data_path):
    """
    Sequentially runs all hyperparameter combinations defined in the config file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        data_path (str): Path to the dataset.
    """
    # Load configuration.
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    
    search_space = config_data.get('grid_search', {})
    
    spatial_sizes = search_space.get('spatial_size', [])
    batch_sizes = search_space.get('batch_size', [])
    lrs = search_space.get('lr', [])

    # Generate all hyperparameter combinations.
    combinations = list(itertools.product(spatial_sizes, batch_sizes, lrs))

    print(f"Starting Sequential Scheduler. Total tasks found: {len(combinations)}")
    print("-" * 50)

    for i, (size, batch, lr) in enumerate(combinations):
        # Construct Paths and Command.
        size_str = "x".join(map(str, size))
        exp_name = f"exp_{size_str}_bs{batch}_lr{lr}"

        # Save checkpoints and logs under this directory.
        save_dir = os.path.join("./checkpoints", exp_name)
        log_file = os.path.join(save_dir, f"{exp_name}.log")
        
        # Ensure log directory exists.
        os.makedirs(save_dir, exist_ok=True)
        
        cmd = [
            "python", "train.py",
            "--data_path", data_path,
            "--save_dir", save_dir,
            "--crop_size", str(size[0]), str(size[1]), str(size[2]),
            "--batch_size", str(batch),
            "--lr", str(lr),
            "--epochs", "100",
        ]
        
        print(f"[{i+1}/{len(combinations)}] Starting: {exp_name}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Log will be saved to: {log_file}")

        # Execute Task Sequentially.
        start_time = time.time()
        
        try:
            with open(log_file, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            
            elapsed_time = time.time() - start_time
            print(f"[SUCCESS] {exp_name} completed in {elapsed_time:.2f} seconds.")
            
        except subprocess.CalledProcessError as e:
            print(f"[FAILURE] {exp_name} failed. Exit code: {e.returncode}. Check log at {log_file}")
        
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred for {exp_name}: {e}")
            
        print("-" * 50)

# ==========================================
# Main Execution Block.
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Grid Search Experiment Runner")

    parser.add_argument('--config', type=str, default='./configs/config.yaml', help="Path to config file")
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    
    args = parser.parse_args()
    run_experiment(args.config, args.data_path)