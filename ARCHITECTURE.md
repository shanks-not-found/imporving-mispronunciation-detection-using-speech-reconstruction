# System Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MISPRONUNCIATION DETECTION SYSTEM                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  Raw Audio Waveform (16kHz)                                         │
│  Shape: (batch_size, sequence_length)                               │
└─────────────┬───────────────────────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                               │
├─────────────────────────────────────────────────────────────────────┤
│  Wav2Vec2 Pre-trained Model                                         │
│  - Convolutional feature encoder                                    │
│  - Transformer encoder (12 layers)                                  │
│  - Contextualized representations                                   │
│  Output Shape: (batch_size, time_steps, 768)                        │
└─────────────┬───────────────────────────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXTUAL ENCODER                               │
├─────────────────────────────────────────────────────────────────────┤
│  Transformer Encoder (4 layers)                                     │
│  - Multi-head attention (8 heads)                                   │
│  - Feed-forward networks (3072 dim)                                 │
│  - Layer normalization                                              │
│  - Residual connections                                             │
│  Output Shape: (batch_size, time_steps, 768)                        │
└─────────────┬───────────────────────────────────────────────────────┘
              │
              ├──────────────────┬──────────────────┐
              ↓                  ↓                  ↓
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  DETECTION HEAD   │  │ RECONSTRUCTION    │  │  HIDDEN STATES    │
│                   │  │ HEAD              │  │                   │
│  Mean Pooling     │  │  Linear Projection│  │  For Analysis     │
│  ↓                │  │  ↓                │  │  & Visualization  │
│  Linear (768→768) │  │  Conv1d Postnet   │  │                   │
│  ↓                │  │  ↓                │  │                   │
│  Tanh             │  │  Residual Add     │  │                   │
│  ↓                │  │  ↓                │  │                   │
│  Dropout (0.1)    │  │  Output           │  │                   │
│  ↓                │  │                   │  │                   │
│  Linear (768→2)   │  │                   │  │                   │
│  ↓                │  │                   │  │                   │
│  Softmax          │  │                   │  │                   │
└────────┬──────────┘  └────────┬──────────┘  └───────────────────┘
         │                      │
         ↓                      ↓
┌───────────────────┐  ┌───────────────────┐
│ CLASSIFICATION    │  │ MEL SPECTROGRAM   │
│                   │  │                   │
│ [Correct,         │  │ Shape:            │
│  Mispronounced]   │  │ (batch, time, 80) │
└────────┬──────────┘  └────────┬──────────┘
         │                      │
         └──────────┬───────────┘
                    ↓
         ┌─────────────────────┐
         │   JOINT LOSS        │
         ├─────────────────────┤
         │ α × Detection Loss  │
         │ + β × Recon Loss    │
         └─────────────────────┘
```

## Detailed Component Flow

### 1. Input Processing
```
Audio File (.wav, .mp3, .flac)
    ↓
Load with torchaudio
    ↓
Resample to 16kHz
    ↓
Convert stereo → mono
    ↓
Pad/Trim to max_length
    ↓
Create attention mask
    ↓
Ready for model
```

### 2. Feature Extraction (Wav2Vec2)
```
Input: (B, T) waveform
    ↓
Conv layers (7 layers, 512 channels)
    ↓
Quantization module
    ↓
Transformer (12 layers, 768 dim)
    ↓
Output: (B, T', 768) features
```

### 3. Transformer Encoder
```
Input: (B, T', 768)
    ↓
For each layer (4 times):
    Multi-Head Attention (8 heads)
    ↓
    Add & Norm
    ↓
    Feed Forward (768→3072→768)
    ↓
    Add & Norm
    ↓
Output: (B, T', 768)
```

### 4A. Detection Path
```
Input: (B, T', 768)
    ↓
Mean pooling (with attention mask)
    ↓
Linear(768→768) + Tanh
    ↓
Dropout(0.1)
    ↓
Linear(768→2)
    ↓
Softmax
    ↓
Output: (B, 2) probabilities
```

### 4B. Reconstruction Path
```
Input: (B, T', 768)
    ↓
Linear projection (768→80)
    ↓
Conv1d postnet (5 layers)
    - Conv(80→256), BatchNorm, Tanh
    - Conv(256→80), BatchNorm, Tanh
    ↓
Residual connection
    ↓
Output: (B, T', 80) mel spectrogram
```

## Training Flow

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                        │
└─────────────────────────────────────────────────────────┘

For each epoch:
    For each batch:
        1. Load audio + labels
        2. Forward pass
        3. Compute losses:
           - Detection: CrossEntropy(pred, label)
           - Reconstruction: MSE(pred_mel, target_mel)
           - Total: α × det_loss + β × rec_loss
        4. Backward pass
        5. Gradient clipping (max_norm=1.0)
        6. Optimizer step (every N accumulation steps)
        7. Log metrics to TensorBoard
    
    Validation:
        - Evaluate on val set
        - Compute metrics (Acc, F1, etc.)
        - Save best checkpoint
```

## Inference Flow

```
┌─────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                    │
└─────────────────────────────────────────────────────────┘

Input: Audio file path
    ↓
Load & preprocess audio
    ↓
Load trained model
    ↓
Forward pass (no gradient)
    ↓
Get detection logits
    ↓
Apply softmax → probabilities
    ↓
Argmax → predicted class
    ↓
Optionally: Save reconstructed mel
    ↓
Return: {
    prediction: "Correct" | "Mispronounced",
    confidence: float,
    probabilities: dict,
    reconstructed_mel: array (optional)
}
```

## Data Format

### Training Data
```json
{
  "audio_path": "path/to/audio.wav",
  "label": 0,  // 0=correct, 1=mispronounced
  "text": "transcription",
  "reference_audio": "path/to/reference.wav"
}
```

### Model Output
```python
{
  "detection_logits": Tensor(B, 2),
  "reconstruction_output": Tensor(B, T, 80),
  "loss": Tensor(1),
  "detection_loss": Tensor(1),
  "reconstruction_loss": Tensor(1),
  "hidden_states": Tensor(B, T, 768)
}
```

## Key Parameters

### Model Architecture
- Hidden size: 768
- Transformer layers: 4
- Attention heads: 8
- Feedforward dim: 3072
- Dropout: 0.1

### Training
- Batch size: 16
- Learning rate: 1e-5
- Epochs: 50
- Optimizer: AdamW
- Scheduler: Linear warmup

### Loss Weights
- Detection weight (α): 0.7
- Reconstruction weight (β): 0.3

### Audio
- Sample rate: 16,000 Hz
- Max length: 160,000 samples (10 sec)
- Mel bins: 80
- FFT size: 1024
- Hop length: 256

## File Dependencies

```
train.py
  ├─ src/models/mispronunciation_model.py
  │   └─ transformers.Wav2Vec2Model
  ├─ src/data/data_loader.py
  │   ├─ torchaudio
  │   └─ librosa
  └─ src/utils/helpers.py
      ├─ sklearn.metrics
      └─ tensorboard

inference.py
  ├─ src/models/mispronunciation_model.py
  └─ src/data/data_loader.py
```

## GPU Memory Usage

| Configuration | Memory | Training Speed |
|--------------|--------|----------------|
| Batch 16, frozen backbone | ~8 GB | Fast |
| Batch 16, full training | ~12 GB | Medium |
| Batch 32, frozen backbone | ~14 GB | Faster |
| Batch 32, full training | ~20 GB | Slower |

## Performance Metrics

Expected results on CAPT datasets:
- Accuracy: 85-95%
- Precision: 0.80-0.92
- Recall: 0.80-0.90
- F1 Score: 0.80-0.92
- Training time: 4-8 hours on GPU (50 epochs, 10k samples)
