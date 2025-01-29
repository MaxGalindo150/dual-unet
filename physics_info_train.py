import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse

from unet_model import UNet
from attention_unet_model import AttentionUNet
from physics_informed import PhysicsInformedWrapper, PATLoss
from preprocess.preprocess_simulated_data import load_and_preprocess_data

def setup_training(alpha):
    """Configura el criterio de pérdida, optimizador y scheduler"""
    return PATLoss(alpha=alpha), optim.Adam, optim.lr_scheduler.ReduceLROnPlateau

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entrena el modelo por una época."""
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc='Training') as t:
        for data, target in t:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            refined, simulated, uncertainty = model(data)
            loss, img_loss, phys_loss = criterion((refined, simulated, uncertainty), (target, data))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            t.set_postfix({'loss': loss.item(), 'img_loss': img_loss.item(), 'phys_loss': phys_loss.item()})
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Valida el modelo por una época."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            refined, simulated, uncertainty = model(data)
            loss, _, _ = criterion((refined, simulated, uncertainty), (target, data))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, num_epochs, stage, alpha, lr, device='cuda'):
    """Entrena el modelo en la fase especificada."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'{stage}_results_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    criterion, optimizer_cls, scheduler_cls = setup_training(alpha)
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    scheduler = scheduler_cls(optimizer, mode='min', patience=5, factor=0.5)
    writer = SummaryWriter(f'runs/photoacoustic_{stage}_{timestamp}')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        writer.add_scalar(f'{stage}/Train Loss', train_loss, epoch)
        writer.add_scalar(f'{stage}/Validation Loss', val_loss, epoch)
        
        print(f'{stage} Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_{stage}_model.pth')
    
        if (epoch + 1) % 10 == 0:
            visualize_results(model, val_loader, device, epoch, save_dir, phase=stage)

    return model, save_dir



def pretrain_model(model, train_loader, val_loader, num_epochs, device='cuda'):
    return train_model(model, train_loader, val_loader, num_epochs, stage='pretrain', alpha=1.0, lr=0.001, device=device)

def finetune_model(model, train_loader, val_loader, num_epochs, device='cuda'):
    return train_model(model, train_loader, val_loader, num_epochs, stage='finetune', alpha=0.7, lr=0.0001, device=device)

def visualize_results(model, val_loader, device, epoch, save_dir, phase='pretrain'):
    """Visualiza resultados durante el entrenamiento"""
    model.eval()
    with torch.no_grad():
        sample_data, sample_target = next(iter(val_loader))
        sample_data = sample_data.to(device)
        
        if phase == 'pretrain-no':
            sample_output = model(sample_data)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(sample_data.cpu().numpy()[0, 0])
            axes[0].set_title('Input')
            axes[1].imshow(sample_target.numpy()[0, 0])
            axes[1].set_title('Ground Truth')
            axes[2].imshow(sample_output.cpu().numpy()[0, 0])
            axes[2].set_title('Prediction')
        else:
            refined, simulated, uncertainty = model(sample_data)
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            axes[0,0].imshow(sample_data.cpu().numpy()[0, 0])
            axes[0,0].set_title('Input')
            axes[0,1].imshow(sample_target.numpy()[0, 0])
            axes[0,1].set_title('Ground Truth')
            axes[1,0].imshow(refined.cpu().numpy()[0, 0])
            axes[1,0].set_title('Refined Prediction')
            axes[1,1].imshow(uncertainty.cpu().numpy()[0, 0])
            axes[1,1].set_title('Uncertainty Map')
            
        
        plt.savefig(f'{save_dir}/{phase}_epoch_{epoch+1}_samples.png')
        plt.close()

def train_dgpinn(model_name, train_loader, val_loader, pretrain_epochs=100, finetune_epochs=50, device='cuda', pretrain_dir=None):
    """Proceso completo de entrenamiento DG-PINN."""
    if pretrain_dir is None:
        model_cls = UNet if model_name == 'unet' else AttentionUNet
        base_model = PhysicsInformedWrapper(model_cls(in_channels=1, out_channels=1)).to(device)
        print("Starting pre-training phase...")
        base_model, pretrain_dir = pretrain_model(base_model, train_loader, val_loader, pretrain_epochs, device)
    else:
        model_path = os.path.join(pretrain_dir, 'best_pretrain_model.pth')
        model_cls = UNet if model_name == 'unet' else AttentionUNet
        base_model = model_cls(in_channels=1, out_channels=1).to(device)
        print(f"Loading pretrained model from {model_path}")
        base_model.load_state_dict(torch.load(model_path))
    
    print("Starting fine-tuning phase...")
    # finetuned_model = PhysicsInformedWrapper(base_model).to(device)
    finetuned_model, finetune_dir = finetune_model(base_model, train_loader, val_loader, finetune_epochs, device)
    
    return finetuned_model, (pretrain_dir, finetune_dir)

def main():
    parser = argparse.ArgumentParser(description="Train a DG-PINN model for photoacoustic image reconstruction.")
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'attention_unet'],
                      help="Name of the base model to train")
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                      help="Number of epochs for pre-training phase")
    parser.add_argument('--finetune_epochs', type=int, default=50,
                      help="Number of epochs for fine-tuning phase")
    parser.add_argument('--pretrain_dir', type=str, default=None,
                      help="Directory with pre-trained model")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = load_and_preprocess_data("simulated_data")
    
    model, save_dirs = train_dgpinn(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        device=device,
        pretrain_dir=args.pretrain_dir
    )

if __name__ == "__main__":
    main()