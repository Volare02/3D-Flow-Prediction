import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 假设 src 文件夹在当前目录
from src.dataset import FlowDataset
from src.models import HybridUNet

# ==========================================
# 1. Configuration.
# ==========================================
DEFAULT_CONFIG = {
    "experiment_name": "TransUNet_MSE",
    "data_path": "/data1/wangteng/processed/channel_flow.hdf5",
    "checkpoint_dir": "./checkpoints",
    "log_dir": "./logs",
    
    # Hyperparameters.
    "batch_size": 16,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "num_workers": 8,
    "spatial_crop_size": (64, 64, 64),
    "dt_per_frame": 0.05, 
}

def parse_args():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description="3D Flow Prediction Training")
    
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints and logs')
    
    parser.add_argument('--crop_size', type=int, nargs=3, help='Spatial crop size (D H W)')
    
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    
    return parser.parse_args()

# ==========================================
# 2. Training Engine.
# ==========================================
def train(config):
    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Experiment: {config['experiment_name']}")
    print(f"   Device: {device}")
    print(f"   Config: Size={config['spatial_crop_size']}, BS={config['batch_size']}, LR={config['learning_rate']}")
    
    # Create Directories.
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Initialize CSV Logger.
    log_csv_path = os.path.join(config["log_dir"], "training_log.csv")
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Learning_Rate'])

    # 1. Dataset & DataLoader.
    print("Initializing DataLoaders...")
    
    train_dataset = FlowDataset(
        file_path=config["data_path"], 
        mode="train", 
        spatial_size=config["spatial_crop_size"],
        dt_per_frame=config["dt_per_frame"]
    )
    val_dataset = FlowDataset(
        file_path=config["data_path"], 
        mode="val", 
        spatial_size=config["spatial_crop_size"], 
        dt_per_frame=config["dt_per_frame"]
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"], 
        pin_memory=True
    )

    # 2. Model.
    print("Initializing Model...")
    model = HybridUNet(in_channels=4, out_channels=4, base_dim=32, time_emb_dim=128).to(device)
    
    # 3. Optimizer & Scheduler.
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    # 4. Loss & Scaler.
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler()

    # ========== Training Loop ==========
    best_val_loss = float("inf")
    
    for epoch in range(config["num_epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", mininterval=10)
        
        train_loss_sum = 0.0
        for inputs, targets, dt in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            dt = dt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast():
                preds = model(inputs, dt)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            pbar.set_postfix({"MSE": f"{loss.item():.6f}"})

        avg_train_loss = train_loss_sum / len(train_loader)

        # ========== Validation ==========
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs, targets, dt in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                dt = dt.to(device, non_blocking=True)
                
                with torch.amp.autocast():
                    preds = model(inputs, dt)
                    loss = criterion(preds, targets)
                
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ========== LOGGING ==========
        print(f"   [Summary] Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, current_lr])

        # Save Checkpoints.
        # 1. Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            
        # 2. Latest Model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, os.path.join(config["checkpoint_dir"], "latest.pth"))

if __name__ == "__main__":
    config = DEFAULT_CONFIG.copy()
    
    args = parse_args()
    
    if args.data_path:
        config["data_path"] = args.data_path
        
    if args.save_dir:
        config["checkpoint_dir"] = args.save_dir
        config["log_dir"] = os.path.join(args.save_dir, "logs")
        config["experiment_name"] = os.path.basename(args.save_dir)
        
    if args.crop_size:
        config["spatial_crop_size"] = tuple(args.crop_size)
        
    if args.batch_size:
        config["batch_size"] = args.batch_size
        
    if args.lr:
        config["learning_rate"] = args.lr
        
    if args.epochs:
        config["num_epochs"] = args.epochs

    train(config)