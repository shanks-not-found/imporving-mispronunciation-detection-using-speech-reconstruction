"""
Training script for mispronunciation detection and speech reconstruction model.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.mispronunciation_model import MispronunciationDetectionModel
from src.data.data_loader import (
    AudioProcessor, 
    load_dataset_from_json, 
    create_dataloaders,
    create_sample_dataset
)
from src.utils.helpers import (
    load_config, 
    save_checkpoint, 
    load_checkpoint,
    compute_metrics,
    AverageMeter,
    get_lr,
    format_time,
    count_parameters,
    set_seed
)


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, writer, args):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    detection_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    
    predictions = []
    targets = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        target_mel = batch['target_mel'].to(device)
        
        # Forward pass
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            target_mel=target_mel
        )
        
        loss = outputs['loss']
        detection_loss = outputs['detection_loss']
        reconstruction_loss = outputs['reconstruction_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # Update weights
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update metrics
        losses.update(loss.item(), input_values.size(0))
        detection_losses.update(detection_loss.item(), input_values.size(0))
        reconstruction_losses.update(reconstruction_loss.item(), input_values.size(0))
        
        # Get predictions
        logits = outputs['detection_logits']
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'det_loss': f'{detection_losses.avg:.4f}',
            'rec_loss': f'{reconstruction_losses.avg:.4f}',
            'lr': f'{get_lr(optimizer):.2e}'
        })
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + step
        if step % args.logging_steps == 0:
            writer.add_scalar('train/loss', losses.avg, global_step)
            writer.add_scalar('train/detection_loss', detection_losses.avg, global_step)
            writer.add_scalar('train/reconstruction_loss', reconstruction_losses.avg, global_step)
            writer.add_scalar('train/learning_rate', get_lr(optimizer), global_step)
    
    # Compute metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    metrics = compute_metrics(predictions, targets)
    
    return losses.avg, metrics


def validate(model, val_loader, device, epoch, writer):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    detection_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move batch to device
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            target_mel = batch['target_mel'].to(device)
            
            # Forward pass
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
                target_mel=target_mel
            )
            
            loss = outputs['loss']
            detection_loss = outputs['detection_loss']
            reconstruction_loss = outputs['reconstruction_loss']
            
            # Update metrics
            losses.update(loss.item(), input_values.size(0))
            detection_losses.update(detection_loss.item(), input_values.size(0))
            reconstruction_losses.update(reconstruction_loss.item(), input_values.size(0))
            
            # Get predictions
            logits = outputs['detection_logits']
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # Compute metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    metrics = compute_metrics(predictions, targets)
    
    # Log to tensorboard
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/detection_loss', detection_losses.avg, epoch)
    writer.add_scalar('val/reconstruction_loss', reconstruction_losses.avg, epoch)
    writer.add_scalar('val/accuracy', metrics['accuracy'], epoch)
    writer.add_scalar('val/f1_score', metrics['f1_score'], epoch)
    
    return losses.avg, metrics


def main(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Merge config with args
    model_config = {
        'wav2vec2_model': config['model']['wav2vec2_model'],
        'hidden_size': config['model']['hidden_size'],
        'num_transformer_layers': config['model']['num_transformer_layers'],
        'num_attention_heads': config['model']['num_attention_heads'],
        'intermediate_size': config['model']['intermediate_size'],
        'dropout': config['model']['dropout'],
        'num_pronunciation_classes': config['model']['num_pronunciation_classes'],
        'mel_bins': config['reconstruction']['mel_bins'],
        'target_length': config['reconstruction']['target_length'],
        'detection_weight': config['reconstruction']['detection_weight'],
        'reconstruction_weight': config['reconstruction']['reconstruction_weight']
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = MispronunciationDetectionModel(model_config)
    model = model.to(device)
    
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Optionally freeze wav2vec2
    if args.freeze_wav2vec2:
        print("Freezing Wav2Vec2 backbone...")
        model.freeze_wav2vec2()
    
    # Prepare data
    print("Preparing data...")
    audio_processor = AudioProcessor(
        sample_rate=config['data']['sample_rate'],
        max_length=int(config['data']['max_audio_length'] * config['data']['sample_rate'])
    )
    
    # Check if dataset exists, otherwise create sample
    if not os.path.exists(args.train_data):
        print("Dataset not found. Creating sample dataset configuration...")
        create_sample_dataset('./data/sample')
        print("Please provide actual audio files and update the dataset JSON.")
        return
    
    # Load datasets
    train_data = load_dataset_from_json(args.train_data)
    val_data = load_dataset_from_json(args.val_data) if args.val_data else []
    test_data = load_dataset_from_json(args.test_data) if args.test_data else []
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        audio_processor,
        batch_size=config['training']['batch_size'],
        mel_bins=config['reconstruction']['mel_bins'],
        target_mel_length=config['reconstruction']['target_length']
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps'],
        weight_decay=config['optimizer']['weight_decay']
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * config['training']['num_epochs'] // config['training']['gradient_accumulation_steps']
    warmup_steps = config['training']['warmup_steps']
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps
    )
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Load checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_metric']
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, writer, args
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Train F1: {train_metrics['f1_score']:.4f}")
        
        # Validate
        if len(val_data) > 0 and (epoch + 1) % args.eval_every == 0:
            val_loss, val_metrics = validate(model, val_loader, device, epoch, writer)
            
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            
            # Save best model
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 0, best_f1,
                    config['training']['checkpoint_dir'], 'best_model.pt'
                )
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, 0, best_f1,
                config['training']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pt'
            )
    
    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mispronunciation detection model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--train_data', type=str, default='data/sample/dataset.json',
                        help='Path to training data JSON')
    parser.add_argument('--val_data', type=str, default='',
                        help='Path to validation data JSON')
    parser.add_argument('--test_data', type=str, default='',
                        help='Path to test data JSON')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--freeze_wav2vec2', action='store_true',
                        help='Freeze Wav2Vec2 backbone')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Tensorboard log directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)
