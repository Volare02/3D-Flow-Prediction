import os
import yaml
import itertools
import subprocess
import argparse
import time
import threading

# ==========================================
# 1. GPU Memory Cost Estimator
# ==========================================
def estimate_gpu_cost(spatial_size, batch_size):
    """
    Estimate the GPU memory cost score for a task.
    Formula: D * H * W * BatchSize / Base_Unit.
    """
    d, h, w = spatial_size
    # Empirical base unit for normalization.
    # Assumption: 64^3 * 16 (BS) consumes roughly 1 unit of memory score.
    base_unit = 64 * 64 * 64 * 16 
    
    total_pixels = d * h * w * batch_size
    cost = total_pixels / base_unit
    
    return cost + 0.5

# ==========================================
# 2. Thread-Safe GPU Resource Manager.
# ==========================================
class GPUResourceManager:
    def __init__(self, total_capacity):
        self.total_capacity = total_capacity
        self.current_usage = 0.0
        self.lock = threading.Condition()

    def acquire(self, cost):
        """
        Request resource. Blocks if insufficient capacity.
        """
        with self.lock:
            # Wait while requested resource exceeds available capacity.
            while self.current_usage + cost > self.total_capacity:
                self.lock.wait()
            # Allocate resource.
            self.current_usage += cost
            print(f"[Allocated] Cost: {cost:.2f} | Current Load: {self.current_usage:.2f}/{self.total_capacity}")

    def release(self, cost):
        """
        Release resource after task completion.
        """
        with self.lock:
            self.current_usage -= cost
            print(f"[Released] Cost: {cost:.2f} | Current Load: {self.current_usage:.2f}/{self.total_capacity}")
            self.lock.notify_all()

# ==========================================
# 3. Worker Thread Function.
# ==========================================
def worker_thread(gpu_manager, cmd, exp_name, log_dir, cost):
    # 1. Acquire GPU resource (Blocking).
    gpu_manager.acquire(cost)
    
    # 2. Run the experiment task.
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    
    print(f"[START] {exp_name}")
    try:
        with open(log_file, "w") as f:
            # Execute train.py as a subprocess.
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"[DONE] {exp_name}")
    except Exception as e:
        print(f"[FAIL] {exp_name} - Check log at {log_file}")
    finally:
        # 3. Always release resources upon completion or failure.
        gpu_manager.release(cost)

# ==========================================
# 4. Main Scheduler Logic.
# ==========================================
def run_experiment(config_path, data_path, gpu_capacity, dry_run=False):
    # Load configuration.
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    search_space = config_data['grid_search']
    
    # Generate all hyperparameter combinations.
    combinations = list(itertools.product(
        search_space['spatial_size'],
        search_space['batch_size'],
        search_space['lr']
    ))

    gpu_manager = GPUResourceManager(total_capacity=gpu_capacity)
    threads = []

    print(f"Starting Smart Scheduler. Total GPU Capacity Score: {gpu_capacity}")
    print(f"Total tasks found: {len(combinations)}")

    for size, batch, lr in combinations:
        # Estimate cost for the current task.
        cost = estimate_gpu_cost(size, batch)
        
        # Skip if a single task exceeds total GPU capacity.
        if cost > gpu_capacity:
            print(f"[SKIP] Task too large! Cost {cost:.2f} > Capacity {gpu_capacity}. ({size}, BS={batch})")
            continue

        # Construct experiment name and paths.
        size_str = "x".join(map(str, size))
        exp_name = f"exp_{size_str}_bs{batch}_lr{lr}"
        save_dir = os.path.join("./checkpoints", exp_name)
        
        cmd = [
            "python", "train.py",
            "--data_path", data_path,
            "--save_dir", save_dir,
            "--crop_size", str(size[0]), str(size[1]), str(size[2]),
            "--batch_size", str(batch),
            "--lr", str(lr),
            "--epochs", "100",
        ]
        
        if dry_run:
            print(f"  [Dry Run] Cost: {cost:.2f} | Command: {' '.join(cmd)}")
            continue

        # Start a worker thread for this task.
        t = threading.Thread(target=worker_thread, args=(gpu_manager, cmd, exp_name, save_dir, cost))
        t.start()
        threads.append(t)
        
        # Small sleep to prevent burst I/O requests.
        time.sleep(1)

    # Wait for all threads to complete.
    for t in threads:
        t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Grid Search with GPU Resource Management")
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help="Path to config file")
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--gpu_capacity', type=float, default=14.0, help='Total GPU capacity score limit')
    parser.add_argument('--dry_run', action='store_true', help="Print commands without executing")
    
    args = parser.parse_args()
    
    run_experiment(args.config, args.data_path, args.gpu_capacity, args.dry_run)