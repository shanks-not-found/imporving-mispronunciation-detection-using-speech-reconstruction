"""
Visualization utilities for mispronunciation detection.
This module provides functions to visualize audio, mel spectrograms, and predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from typing import Optional, Tuple
import torch


def plot_waveform(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    title: str = "Waveform",
    save_path: Optional[str] = None
):
    """
    Plot audio waveform.
    
    Args:
        waveform: Audio waveform (1D array)
        sample_rate: Sample rate in Hz
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 4))
    time = np.arange(len(waveform)) / sample_rate
    plt.plot(time, waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_mel_spectrogram(
    mel_spec: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 256,
    title: str = "Mel Spectrogram",
    save_path: Optional[str] = None
):
    """
    Plot mel spectrogram.
    
    Args:
        mel_spec: Mel spectrogram (2D array: time x mel_bins)
        sample_rate: Sample rate in Hz
        hop_length: Hop length for time axis
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 6))
    
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.detach().cpu().numpy()
    
    # Transpose if needed (librosa expects freq x time)
    if mel_spec.shape[0] > mel_spec.shape[1]:
        mel_spec = mel_spec.T
    
    librosa.display.specshow(
        mel_spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comparison(
    original_mel: np.ndarray,
    reconstructed_mel: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 256,
    save_path: Optional[str] = None
):
    """
    Plot comparison between original and reconstructed mel spectrograms.
    
    Args:
        original_mel: Original mel spectrogram
        reconstructed_mel: Reconstructed mel spectrogram
        sample_rate: Sample rate in Hz
        hop_length: Hop length for time axis
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Prepare data
    if isinstance(original_mel, torch.Tensor):
        original_mel = original_mel.detach().cpu().numpy()
    if isinstance(reconstructed_mel, torch.Tensor):
        reconstructed_mel = reconstructed_mel.detach().cpu().numpy()
    
    # Transpose if needed
    if original_mel.shape[0] > original_mel.shape[1]:
        original_mel = original_mel.T
    if reconstructed_mel.shape[0] > reconstructed_mel.shape[1]:
        reconstructed_mel = reconstructed_mel.T
    
    # Plot original
    librosa.display.specshow(
        original_mel,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis',
        ax=axes[0]
    )
    axes[0].set_title('Original Mel Spectrogram')
    fig.colorbar(axes[0].collections[0], ax=axes[0], format='%+2.0f dB')
    
    # Plot reconstructed
    librosa.display.specshow(
        reconstructed_mel,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis',
        ax=axes[1]
    )
    axes[1].set_title('Reconstructed Mel Spectrogram')
    fig.colorbar(axes[1].collections[0], ax=axes[1], format='%+2.0f dB')
    
    # Plot difference
    min_len = min(original_mel.shape[1], reconstructed_mel.shape[1])
    difference = original_mel[:, :min_len] - reconstructed_mel[:, :min_len]
    
    im = librosa.display.specshow(
        difference,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='coolwarm',
        ax=axes[2]
    )
    axes[2].set_title('Difference (Original - Reconstructed)')
    fig.colorbar(im, ax=axes[2], format='%+2.0f dB')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_prediction_confidence(
    probabilities: dict,
    title: str = "Prediction Confidence",
    save_path: Optional[str] = None
):
    """
    Plot prediction confidence as a bar chart.
    
    Args:
        probabilities: Dictionary with class probabilities
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(8, 6))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ['green' if p < 0.5 else 'red' for p in probs]
    
    bars = plt.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Probability')
    plt.title(title)
    plt.ylim(0, 1)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{prob:.4f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    title: str = "Attention Weights",
    save_path: Optional[str] = None
):
    """
    Plot attention weights heatmap.
    
    Args:
        attention_weights: Attention weight matrix (2D)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
):
    """
    Plot training history with multiple metrics.
    
    Args:
        history: Dictionary with training metrics
            - 'train_loss': list of training losses
            - 'val_loss': list of validation losses
            - 'train_acc': list of training accuracies
            - 'val_acc': list of validation accuracies
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Plot training and validation loss
    if 'train_loss' in history:
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training')
    if 'val_acc' in history:
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot detection loss
    if 'train_det_loss' in history:
        axes[1, 0].plot(epochs, history['train_det_loss'], 'b-', label='Training')
    if 'val_det_loss' in history:
        axes[1, 0].plot(epochs, history['val_det_loss'], 'r-', label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Detection Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot reconstruction loss
    if 'train_rec_loss' in history:
        axes[1, 1].plot(epochs, history['train_rec_loss'], 'b-', label='Training')
    if 'val_rec_loss' in history:
        axes[1, 1].plot(epochs, history['val_rec_loss'], 'r-', label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Reconstruction Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_batch_predictions(
    predictions: list,
    labels: list,
    confidences: list,
    save_path: Optional[str] = None
):
    """
    Visualize batch prediction results.
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        confidences: List of confidence scores
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion-style visualization
    correct = [p == l for p, l in zip(predictions, labels)]
    colors = ['green' if c else 'red' for c in correct]
    
    x = range(len(predictions))
    ax1.scatter(x, predictions, c=colors, alpha=0.6, s=100, label='Predictions')
    ax1.scatter(x, labels, marker='x', c='blue', s=50, label='True Labels')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Class (0=Correct, 1=Mispronounced)')
    ax1.set_title('Predictions vs True Labels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confidence distribution
    ax2.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print accuracy
    accuracy = sum(correct) / len(correct)
    print(f"Batch Accuracy: {accuracy:.4f} ({sum(correct)}/{len(correct)})")
