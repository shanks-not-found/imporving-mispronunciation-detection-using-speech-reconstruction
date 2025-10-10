"""
Complete End-to-End Example
This script demonstrates the complete workflow from data preparation to inference.
"""

import os
import json
import numpy as np
import torch
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("END-TO-END MISPRONUNCIATION DETECTION EXAMPLE")
print("=" * 80)

print("\n[STEP 1] Setup and Configuration")
print("-" * 80)

# Import after adding to path
from src.utils.helpers import load_config, set_seed
from src.models.mispronunciation_model import MispronunciationDetectionModel
from src.data.data_loader import AudioProcessor

# Set random seed for reproducibility
set_seed(42)
print("✓ Random seed set to 42")

# Load configuration
config = load_config('configs/default_config.yaml')
print("✓ Configuration loaded")
print(f"  - Model: {config['model']['wav2vec2_model']}")
print(f"  - Transformer layers: {config['model']['num_transformer_layers']}")
print(f"  - Attention heads: {config['model']['num_attention_heads']}")
print(f"  - Detection weight: {config['reconstruction']['detection_weight']}")
print(f"  - Reconstruction weight: {config['reconstruction']['reconstruction_weight']}")

print("\n[STEP 2] Model Creation")
print("-" * 80)

# Create model configuration
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

print("Creating model...")
model = MispronunciationDetectionModel(model_config)
print("✓ Model created successfully")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  - Total parameters: {total_params:,}")
print(f"  - Trainable parameters: {trainable_params:,}")

print("\n[STEP 3] Data Preparation")
print("-" * 80)

# Create audio processor
audio_processor = AudioProcessor(
    sample_rate=config['data']['sample_rate'],
    max_length=int(config['data']['max_audio_length'] * config['data']['sample_rate'])
)
print("✓ Audio processor created")
print(f"  - Sample rate: {audio_processor.sample_rate} Hz")
print(f"  - Max length: {audio_processor.max_length} samples")

# Create synthetic audio for demonstration
print("\nCreating synthetic audio data...")
duration = 3.0  # seconds
sample_rate = config['data']['sample_rate']
t = np.linspace(0, duration, int(sample_rate * duration))

# Create two synthetic audio samples
# Sample 1: Simple sine wave (simulating "correct" pronunciation)
freq1 = 440.0  # A4 note
audio1 = np.sin(2 * np.pi * freq1 * t).astype(np.float32)

# Sample 2: Sine wave with noise (simulating "mispronounced")
freq2 = 350.0  # F4 note
audio2 = np.sin(2 * np.pi * freq2 * t).astype(np.float32)
audio2 += np.random.normal(0, 0.1, audio2.shape).astype(np.float32)

print("✓ Synthetic audio created")
print(f"  - Sample 1 (correct): {len(audio1)} samples, {len(audio1)/sample_rate:.2f} seconds")
print(f"  - Sample 2 (mispronounced): {len(audio2)} samples, {len(audio2)/sample_rate:.2f} seconds")

# Process audio
print("\nProcessing audio...")
waveform1 = torch.from_numpy(audio1)
waveform2 = torch.from_numpy(audio2)

waveform1 = audio_processor.pad_or_trim(waveform1)
waveform2 = audio_processor.pad_or_trim(waveform2)

# Extract mel spectrograms
mel1 = audio_processor.extract_mel_spectrogram(
    waveform1, 
    n_mels=config['reconstruction']['mel_bins']
)
mel2 = audio_processor.extract_mel_spectrogram(
    waveform2,
    n_mels=config['reconstruction']['mel_bins']
)

print("✓ Audio processed and mel spectrograms extracted")
print(f"  - Mel spectrogram shape: {mel1.shape}")

print("\n[STEP 4] Forward Pass (Inference)")
print("-" * 80)

# Prepare batch
batch_size = 2
input_values = torch.stack([waveform1, waveform2])
attention_mask = torch.ones(batch_size, len(waveform1))
labels = torch.tensor([0, 1])  # 0=correct, 1=mispronounced

# Pad mels to target length
target_length = config['reconstruction']['target_length']
mel1_padded = torch.nn.functional.pad(mel1, (0, 0, 0, target_length - mel1.size(0)))[:target_length, :]
mel2_padded = torch.nn.functional.pad(mel2, (0, 0, 0, target_length - mel2.size(0)))[:target_length, :]
target_mel = torch.stack([mel1_padded, mel2_padded])

print("Running forward pass...")
model.eval()
with torch.no_grad():
    outputs = model(
        input_values=input_values,
        attention_mask=attention_mask,
        labels=labels,
        target_mel=target_mel
    )

print("✓ Forward pass completed")
print(f"  - Detection logits shape: {outputs['detection_logits'].shape}")
print(f"  - Reconstruction output shape: {outputs['reconstruction_output'].shape}")
print(f"  - Total loss: {outputs['loss'].item():.4f}")
print(f"  - Detection loss: {outputs['detection_loss'].item():.4f}")
print(f"  - Reconstruction loss: {outputs['reconstruction_loss'].item():.4f}")

print("\n[STEP 5] Predictions and Analysis")
print("-" * 80)

# Get predictions
logits = outputs['detection_logits']
probs = torch.softmax(logits, dim=-1)
predictions = torch.argmax(probs, dim=-1)

print("Predictions:")
for i in range(batch_size):
    true_label = labels[i].item()
    pred_label = predictions[i].item()
    confidence = probs[i, pred_label].item()
    
    label_names = ['Correct', 'Mispronounced']
    status = "✓" if true_label == pred_label else "✗"
    
    print(f"\n  Sample {i+1}: {status}")
    print(f"    True Label: {label_names[true_label]}")
    print(f"    Prediction: {label_names[pred_label]}")
    print(f"    Confidence: {confidence:.4f}")
    print(f"    Probabilities:")
    print(f"      - Correct: {probs[i, 0].item():.4f}")
    print(f"      - Mispronounced: {probs[i, 1].item():.4f}")

print("\n[STEP 6] Training Workflow Overview")
print("-" * 80)

print("""
To train this model on real data:

1. Prepare your dataset:
   - Format: JSON with audio paths and labels
   - Example: data/train.json, data/val.json
   
2. Run training:
   python train.py \\
       --config configs/default_config.yaml \\
       --train_data data/train.json \\
       --val_data data/val.json \\
       --log_dir logs/experiment1

3. Monitor training:
   tensorboard --logdir logs/

4. Use trained model:
   python inference.py \\
       --model_path checkpoints/best_model.pt \\
       --config configs/default_config.yaml \\
       --audio_file test.wav
""")

print("\n[STEP 7] Recommended Datasets")
print("-" * 80)

print("""
Public CAPT Datasets for Training:

1. L2-ARCTIC
   - URL: http://psi.engr.tamu.edu/l2-arctic-corpus/
   - Description: Non-native English speech corpus
   - Size: ~27,000 utterances
   
2. speechocean762
   - URL: Available on GitHub/Kaggle
   - Description: Pronunciation scoring dataset
   - Size: ~5,000 utterances
   
3. EpaDB
   - Description: English Pronunciation Assessment Database
   - Contains phoneme-level annotations
""")

print("\n[STEP 8] Model Features")
print("-" * 80)

print("""
Key Features:

✓ Wav2Vec2 Pre-trained Backbone
  - Leverages large-scale pre-training
  - Rich acoustic representations
  
✓ Transformer Encoder
  - Captures long-range dependencies
  - Multi-head self-attention
  
✓ Dual-Task Learning
  - Detection: Binary classification
  - Reconstruction: Mel spectrogram generation
  
✓ Flexible Configuration
  - Easy to customize via YAML
  - Support for different backbones
  
✓ Production Ready
  - Checkpointing and resuming
  - TensorBoard logging
  - Batch inference support
""")

print("\n" + "=" * 80)
print("EXAMPLE COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("""
Next Steps:
1. Review the code in src/ directory
2. Prepare your dataset using public CAPT datasets
3. Customize configs/default_config.yaml
4. Run train.py to start training
5. Use inference.py for predictions

For more information:
- README.md: Complete documentation
- QUICKSTART.md: Quick start guide
- TECHNICAL.md: Technical details
- examples/: More example scripts

Good luck with your mispronunciation detection project! 🚀
""")
