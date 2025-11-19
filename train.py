import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.dataset import FlowDataset
from src.models import HybridUNet 

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "experiment_name": "TransUNet_channel_flow",
    "data_path": "/data1/wangteng/processed/channel_flow.hdf5",
    "grid_path": "/data1/wangteng/raw/channelflow-dns-re544-seq-p000-020/grid",
    "checkpoint_dir": "./checkpoints",
    "log_dir": "./logs",
    
    # Hyperparameters
    "batch_size": 16,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "num_workers": 8,
    "spatial_crop_size": (32, 32, 32),
    
    # Physics Loss Weights
    "lambda_mse": 1.0,
    "lambda_grad": 0.5,
    "lambda_div": 0.1,
}

# ==========================================
# 2. Physics-Informed Loss Function
# ==========================================
class HybridPhysicsLoss(nn.Module):
    """
    Loss = MSE + lambda1 * Gradient_Loss + lambda2 * Divergence_Loss
    """
    def __init__(self, dx=1.0, dy=1.0, dz=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.dx, self.dy, self.dz = dx, dy, dz

    def compute_gradient_3d(self, field):
        """Computes central difference gradients."""
        grad_d = torch.gradient(field, spacing=self.dx, dim=2)[0]
        grad_h = torch.gradient(field, spacing=self.dy, dim=3)[0]
        grad_w = torch.gradient(field, spacing=self.dz, dim=4)[0]
        return grad_d, grad_h, grad_w

    def divergence_loss(self, pred_flow):
        """
        Enforces continuity equation: div(u) = 0.
        """
        u, v, w = pred_flow[:, 0], pred_flow[:, 1], pred_flow[:, 2]
        
        du_dx = torch.gradient(u, spacing=self.dx, dim=1)[0]
        dv_dy = torch.gradient(v, spacing=self.dy, dim=2)[0]
        dw_dz = torch.gradient(w, spacing=self.dz, dim=3)[0]
        
        div = du_dx + dv_dy + dw_dz
        return torch.mean(div ** 2)

    def gradient_loss(self, pred, target):
        """
        Penalizes errors in the derivatives.
        """
        pred_dx, pred_dy, pred_dz = self.compute_gradient_3d(pred)
        target_dx, target_dy, target_dz = self.compute_gradient_3d(target)
        
        loss_grad = self.mse(pred_dx, target_dx) + self.mse(pred_dy, target_dy) + self.mse(pred_dz, target_dz)
        return loss_grad

    def forward(self, preds, targets):
        loss_mse = self.mse(preds, targets)
        loss_grad = self.gradient_loss(preds, targets)
        loss_div = self.divergence_loss(preds[ : , 0 : 3])
        
        return loss_mse, loss_grad, loss_div

# ==========================================
# 3. Training Engine
# ==========================================
def train(config):
    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")
    
    # Create Directories.
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # 1. Dataset & DataLoader.
    print("Initializing DataLoaders...")
    train_dataset = FlowDataset(
        file_path=config["data_path"], 
        mode="train", 
        spatial_crop_size=config["spatial_crop_size"],
        num_spatial_crops=5,
    )
    val_dataset = FlowDataset(
        file_path=config["data_path"], 
        mode="val", 
        spatial_crop_size=config["spatial_crop_size"],
        num_spatial_crops=1,
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, 
                              num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, 
                            num_workers=config["num_workers"], pin_memory=True)

    # 2. Model.
    print("Initializing Model...")
    model = HybridUNet(in_channels=4, out_channels=4, base_dim=32, time_emb_dim=128).to(device)
    
    # 3. Optimizer & Scheduler.
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    # 4. Loss & Scaler.
    criterion = HybridPhysicsLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    # --- Training Loop ---
    best_val_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        epoch_loss = 0
        log_mse, log_grad, log_div = 0, 0, 0
        for inputs, targets, dt in pbar:
            inputs, targets, dt = inputs.to(device), targets.to(device), dt.to(device)
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass.
            with torch.amp.autocast('cuda'):
                preds = model(inputs, dt)
                loss_mse, loss_grad, loss_div = criterion(preds, targets)
                total_loss = (config["lambda_mse"] * loss_mse) + \
                             (config["lambda_grad"] * loss_grad) + \
                             (config["lambda_div"] * loss_div)

            # Backward Pass with Scaler.
            scaler.scale(total_loss).backward()
            
            # Gradient Clipping (Important for Transformer stability).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            epoch_loss += total_loss.item()
            log_mse += loss_mse.item()
            log_grad += loss_grad.item()
            log_div += loss_div.item()
            
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "MSE": f"{loss_mse.item():.4f}",
                "Grad": f"{loss_grad.item():.4f}"
            })

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets, dt in val_loader:
                inputs, targets, dt = inputs.to(device), targets.to(device), dt.to(device)
                
                with torch.amp.autocast('cuda'):
                    preds = model(inputs, dt)
                    l_mse, l_grad, l_div = criterion(preds, targets)
                    total_loss = config["lambda_mse"]*l_mse + config["lambda_grad"]*l_grad + config["lambda_div"]*l_div
                    
                val_loss += total_loss.item()

        # Average Metrics.
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_mse = log_mse / len(train_loader)
        avg_grad = log_grad / len(train_loader)
        avg_div = log_div / len(train_loader)

        # Step Scheduler.
        scheduler.step()

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"  Breakdown : MSE={avg_mse:.6f}, Grad={avg_grad:.6f}, Div={avg_div:.6f}")

        # Save Best Model.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  [Saving] New best model saved to {save_path}")
            
        # Save Latest Model.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, os.path.join(config["checkpoint_dir"], "latest.pth"))

if __name__ == "__main__":
    train(CONFIG)
