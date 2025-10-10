"""
Data preprocessing and loading utilities for mispronunciation detection.
"""

import torch
import torchaudio
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import json
import os


class AudioProcessor:
    """Handles audio loading and preprocessing."""
    
    def __init__(self, sample_rate: int = 16000, max_length: int = 160000):
        self.sample_rate = sample_rate
        self.max_length = max_length  # 10 seconds at 16kHz
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and resample if necessary."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Squeeze to 1D
        waveform = waveform.squeeze(0)
        
        return waveform
    
    def pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or trim waveform to max_length."""
        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]
        elif waveform.shape[0] < self.max_length:
            pad_length = self.max_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform
    
    def extract_mel_spectrogram(
        self, 
        waveform: torch.Tensor, 
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256
    ) -> torch.Tensor:
        """Extract mel spectrogram from waveform."""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        mel_spec = mel_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec.transpose(0, 1)  # (time, mel_bins)


class MispronunciationDataset(Dataset):
    """Dataset for mispronunciation detection and speech reconstruction."""
    
    def __init__(
        self,
        data_list: List[Dict],
        audio_processor: AudioProcessor,
        mel_bins: int = 80,
        target_mel_length: int = 1024
    ):
        """
        Args:
            data_list: List of dictionaries with keys:
                - 'audio_path': path to audio file
                - 'label': 0 (correct) or 1 (mispronounced)
                - 'text': (optional) transcription text
                - 'reference_audio': (optional) path to correct pronunciation
            audio_processor: AudioProcessor instance
            mel_bins: number of mel bins
            target_mel_length: target length for mel spectrogram
        """
        self.data_list = data_list
        self.audio_processor = audio_processor
        self.mel_bins = mel_bins
        self.target_mel_length = target_mel_length
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        
        # Load audio
        waveform = self.audio_processor.load_audio(item['audio_path'])
        
        # Create attention mask (1 for valid tokens, 0 for padding)
        actual_length = min(len(waveform), self.audio_processor.max_length)
        waveform = self.audio_processor.pad_or_trim(waveform)
        
        attention_mask = torch.zeros(len(waveform))
        attention_mask[:actual_length] = 1
        
        # Extract mel spectrogram for reconstruction target
        mel_spec = self.audio_processor.extract_mel_spectrogram(waveform, n_mels=self.mel_bins)
        
        # Pad or trim mel spectrogram
        if mel_spec.shape[0] > self.target_mel_length:
            mel_spec = mel_spec[:self.target_mel_length, :]
        elif mel_spec.shape[0] < self.target_mel_length:
            pad_length = self.target_mel_length - mel_spec.shape[0]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 0, 0, pad_length))
        
        # Get label
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return {
            'input_values': waveform,
            'attention_mask': attention_mask,
            'labels': label,
            'target_mel': mel_spec
        }


def load_dataset_from_json(json_path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_sample_dataset(output_dir: str = './data/sample', num_samples: int = 10):
    """
    Create a sample dataset for demonstration purposes.
    In practice, users should use public CAPT datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample JSON
    sample_data = []
    for i in range(num_samples):
        sample_data.append({
            'audio_path': f'{output_dir}/sample_{i}.wav',
            'label': i % 2,  # Alternate between correct and mispronounced
            'text': f'sample text {i}',
            'reference_audio': f'{output_dir}/reference_{i}.wav'
        })
    
    json_path = os.path.join(output_dir, 'dataset.json')
    with open(json_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample dataset configuration saved to {json_path}")
    print("Note: Audio files need to be provided separately.")
    print("Please use public CAPT datasets like:")
    print("  - L2-ARCTIC")
    print("  - speechocean762")
    print("  - EpaDB")
    
    return json_path


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Stack all tensors
    input_values = torch.stack([item['input_values'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    target_mel = torch.stack([item['target_mel'] for item in batch])
    
    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels,
        'target_mel': target_mel
    }


def create_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    audio_processor: AudioProcessor,
    batch_size: int = 16,
    mel_bins: int = 80,
    target_mel_length: int = 1024
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test dataloaders."""
    
    train_dataset = MispronunciationDataset(
        train_data, audio_processor, mel_bins, target_mel_length
    )
    val_dataset = MispronunciationDataset(
        val_data, audio_processor, mel_bins, target_mel_length
    )
    test_dataset = MispronunciationDataset(
        test_data, audio_processor, mel_bins, target_mel_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
