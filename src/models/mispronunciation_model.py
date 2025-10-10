"""
Mispronunciation Detection and Speech Reconstruction Model

This model jointly trains mispronunciation detection and speech reconstruction
using Wav2Vec2.0 + Transformer layers.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional, Tuple, Dict


class TransformerEncoder(nn.Module):
    """Transformer encoder for processing Wav2Vec2 features."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        """
        if attention_mask is not None:
            # Convert attention mask to transformer format (True = ignore)
            attention_mask = ~attention_mask.bool()
            
        output = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        output = self.layer_norm(output)
        
        return output


class MispronunciationDetectionHead(nn.Module):
    """Classification head for mispronunciation detection."""
    
    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Pool the sequence - mean pooling with attention mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)
            
        logits = self.classifier(pooled)
        return logits


class SpeechReconstructionHead(nn.Module):
    """Reconstruction head for generating corrective speech features."""
    
    def __init__(self, hidden_size: int = 768, mel_bins: int = 80, target_length: int = 1024):
        super().__init__()
        
        self.mel_bins = mel_bins
        self.target_length = target_length
        
        # Upsampling and projection layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, mel_bins)
        )
        
        # Postnet for refinement
        self.postnet = nn.Sequential(
            nn.Conv1d(mel_bins, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(256, mel_bins, kernel_size=5, padding=2),
            nn.BatchNorm1d(mel_bins),
            nn.Tanh(),
            nn.Dropout(0.5)
        )
        
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            mel_outputs: (batch_size, target_length, mel_bins)
        """
        # Project to mel spectrogram space
        mel_outputs = self.projection(hidden_states)  # (batch, seq_len, mel_bins)
        
        # Apply postnet refinement
        mel_transposed = mel_outputs.transpose(1, 2)  # (batch, mel_bins, seq_len)
        postnet_output = self.postnet(mel_transposed)
        mel_outputs = mel_outputs + postnet_output.transpose(1, 2)
        
        return mel_outputs


class MispronunciationDetectionModel(nn.Module):
    """
    Joint model for mispronunciation detection and speech reconstruction.
    
    Architecture:
        1. Wav2Vec2 feature extraction
        2. Transformer encoder layers
        3. Dual heads:
           - Mispronunciation detection (classification)
           - Speech reconstruction (mel spectrogram generation)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config parameters
        wav2vec2_model = config.get('wav2vec2_model', 'facebook/wav2vec2-base')
        hidden_size = config.get('hidden_size', 768)
        num_transformer_layers = config.get('num_transformer_layers', 4)
        num_attention_heads = config.get('num_attention_heads', 8)
        intermediate_size = config.get('intermediate_size', 3072)
        dropout = config.get('dropout', 0.1)
        num_classes = config.get('num_pronunciation_classes', 2)
        mel_bins = config.get('mel_bins', 80)
        target_length = config.get('target_length', 1024)
        
        # Wav2Vec2 backbone
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model)
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            hidden_size=hidden_size,
            num_layers=num_transformer_layers,
            num_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        
        # Task-specific heads
        self.detection_head = MispronunciationDetectionHead(
            hidden_size=hidden_size,
            num_classes=num_classes
        )
        
        self.reconstruction_head = SpeechReconstructionHead(
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            target_length=target_length
        )
        
        # Loss weights
        self.detection_weight = config.get('detection_weight', 0.7)
        self.reconstruction_weight = config.get('reconstruction_weight', 0.3)
        
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_mel: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_values: (batch_size, sequence_length) - raw audio waveform
            attention_mask: (batch_size, sequence_length)
            labels: (batch_size,) - mispronunciation labels
            target_mel: (batch_size, mel_length, mel_bins) - target mel spectrogram
            
        Returns:
            Dictionary containing:
                - detection_logits: mispronunciation classification logits
                - reconstruction_output: reconstructed mel spectrogram
                - loss: combined loss (if labels and target_mel provided)
                - detection_loss: detection loss component
                - reconstruction_loss: reconstruction loss component
        """
        # Extract features with Wav2Vec2
        wav2vec2_outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask
        )
        
        hidden_states = wav2vec2_outputs.last_hidden_state
        
        # Process with transformer encoder
        encoder_output = self.transformer_encoder(hidden_states, attention_mask)
        
        # Get predictions from both heads
        detection_logits = self.detection_head(encoder_output, attention_mask)
        reconstruction_output = self.reconstruction_head(encoder_output)
        
        outputs = {
            'detection_logits': detection_logits,
            'reconstruction_output': reconstruction_output,
            'hidden_states': encoder_output
        }
        
        # Compute losses if labels provided
        if labels is not None and target_mel is not None:
            # Detection loss (cross-entropy)
            detection_loss_fct = nn.CrossEntropyLoss()
            detection_loss = detection_loss_fct(detection_logits, labels)
            
            # Reconstruction loss (MSE on mel spectrogram)
            reconstruction_loss_fct = nn.MSELoss()
            # Align lengths if necessary
            if reconstruction_output.size(1) != target_mel.size(1):
                min_len = min(reconstruction_output.size(1), target_mel.size(1))
                reconstruction_output_aligned = reconstruction_output[:, :min_len, :]
                target_mel_aligned = target_mel[:, :min_len, :]
            else:
                reconstruction_output_aligned = reconstruction_output
                target_mel_aligned = target_mel
                
            reconstruction_loss = reconstruction_loss_fct(
                reconstruction_output_aligned,
                target_mel_aligned
            )
            
            # Combined loss
            total_loss = (
                self.detection_weight * detection_loss +
                self.reconstruction_weight * reconstruction_loss
            )
            
            outputs['loss'] = total_loss
            outputs['detection_loss'] = detection_loss
            outputs['reconstruction_loss'] = reconstruction_loss
            
        return outputs
    
    def freeze_wav2vec2(self):
        """Freeze Wav2Vec2 backbone parameters."""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
    def unfreeze_wav2vec2(self):
        """Unfreeze Wav2Vec2 backbone parameters."""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
