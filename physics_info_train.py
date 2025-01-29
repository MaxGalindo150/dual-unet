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

def pretrain_model(model, train_loader, val_loader, num_epochs, device='cuda'):
    """Primera fase: Pre-entrenamiento usando solo pérdida de datos"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'pretraining_results_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(f'runs/photoacoustic_pretraining_{timestamp}')
    
    # Solo pérdida de reconstrucción
    criterion = PATLoss(alpha=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}/{num_epochs}') as t:
            for batch_idx, (data, target) in enumerate(t):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                refined, simulated, uncertainty = model(data)
                loss, img_loss, phys_loss = criterion(
                    (refined, simulated, uncertainty),
                    (target, data)  # data es la señal original
                )
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                t.set_postfix({
                    'total_loss': loss.item(),
                    'img_loss': img_loss.item(),
                    'phys_loss': phys_loss.item()
                })
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                refined, simulated, uncertainty = model(data)
                loss, _, _ = criterion(
                    (refined, simulated, uncertainty),
                    (target, data)
                )
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_pretrained_model.pth')
        
        writer.add_scalar('PretrainingLoss/train', train_loss, epoch)
        writer.add_scalar('PretrainingLoss/validation', val_loss, epoch)
        
        print(f'Pretrain Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
        if (epoch + 1) % 10 == 0:
            visualize_results(model, val_loader, device, epoch, save_dir, phase='pretrain')
    
    return model, save_dir

def finetune_model(model, train_loader, val_loader, num_epochs, device='cuda'):
    """Segunda fase: Afinamiento con pérdida física"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'finetuning_results_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(f'runs/photoacoustic_finetuning_{timestamp}')
    
    # Pérdida combinada: datos + física
    criterion = PATLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Learning rate más bajo para fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}/{num_epochs}') as t:
            for batch_idx, (data, target) in enumerate(t):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                refined, simulated, uncertainty = model(data)
                loss, img_loss, phys_loss = criterion(
                    (refined, simulated, uncertainty),
                    (target, data)  # data es la señal original
                )
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                t.set_postfix({
                    'total_loss': loss.item(),
                    'img_loss': img_loss.item(),
                    'phys_loss': phys_loss.item()
                })
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                refined, simulated, uncertainty = model(data)
                loss, _, _ = criterion(
                    (refined, simulated, uncertainty),
                    (target, data)
                )
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_finetuned_model.pth')
        
        writer.add_scalar('FinetuningLoss/train', train_loss, epoch)
        writer.add_scalar('FinetuningLoss/validation', val_loss, epoch)
        
        print(f'Finetune Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
        if (epoch + 1) % 10 == 0:
            visualize_results(model, val_loader, device, epoch, save_dir, phase='finetune')
    
    return model, save_dir

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
    """Proceso completo de entrenamiento DG-PINN"""
    
    # Fase 1: Pre-entrenamiento
    if pretrain_dir is None:
        if model_name == 'unet':
            model = UNet(in_channels=1, out_channels=1)
            base_model = PhysicsInformedWrapper(model).to(device)
        elif model_name == 'attention_unet':
            model = AttentionUNet(in_channels=1, out_channels=1).to(device)
            base_model = PhysicsInformedWrapper(model).to(device)
        else:
            raise ValueError("Model name must be 'unet' or 'attention_unet'")
        
        print("Starting pre-training phase...")
        pretrained_model, pretrain_dir = pretrain_model(
            base_model, 
            train_loader, 
            val_loader, 
            num_epochs=pretrain_epochs,
            device=device
        )
        del pretrained_model
    else:
        model_path = pretrain_dir + '/best_pretrained_model.pth'
        if model_name == 'unet':
            pretrained_model = UNet(in_channels=1, out_channels=1).to(device)
        elif model_name == 'attention_unet':
            pretrained_model = AttentionUNet(in_channels=1, out_channels=1).to(device)
        else:
            raise ValueError("Model name must be 'unet' or 'attention_unet'")
        
        print(f"loading params: {model_path}")
        pretrained_model.load_state_dict(torch.load(model_path))
    
    # Fase 2: Afinamiento con física
    print("\nStarting fine-tuning phase...")
    physics_model = PhysicsInformedWrapper(pretrained_model).to(device)
    finetuned_model, finetune_dir = finetune_model(
        physics_model,
        train_loader,
        val_loader,
        num_epochs=finetune_epochs,
        device=device
    )
    
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