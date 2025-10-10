# Technical Documentation

## System Architecture

### Overview

The mispronunciation detection system uses a joint learning approach that combines:
1. **Detection Task**: Binary classification (correct/mispronounced)
2. **Reconstruction Task**: Generate mel spectrogram of correct pronunciation

This dual-task learning improves model performance by:
- Providing richer supervision signal
- Learning better speech representations
- Enabling corrective feedback generation

### Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Audio                          │
│                   (16kHz waveform)                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────┐
│              Wav2Vec2 Feature Extraction                │
│  - Pre-trained on large speech corpus                   │
│  - Outputs contextualized features                      │
│  - Shape: (batch, time, 768)                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────┐
│              Transformer Encoder                        │
│  - 4 layers of multi-head attention                     │
│  - 8 attention heads per layer                          │
│  - Feed-forward dimension: 3072                         │
│  - Dropout: 0.1                                         │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ↓                           ↓
┌──────────────────┐        ┌──────────────────┐
│ Detection Head   │        │ Reconstruction   │
│                  │        │ Head             │
│ - Mean pooling   │        │ - Linear         │
│ - 2 FC layers    │        │   projection     │
│ - Softmax output │        │ - CNN postnet    │
│                  │        │ - Mel output     │
└──────────────────┘        └──────────────────┘
         ↓                           ↓
┌──────────────────┐        ┌──────────────────┐
│ Classification   │        │ Mel Spectrogram  │
│ (Correct/        │        │ (80 bins)        │
│  Mispronounced)  │        │                  │
└──────────────────┘        └──────────────────┘
```

## Model Components

### 1. Wav2Vec2 Backbone

- **Purpose**: Extract rich acoustic features from raw audio
- **Source**: `facebook/wav2vec2-base` (pre-trained)
- **Input**: Raw waveform (16kHz)
- **Output**: Contextualized features (768-dim)
- **Can be frozen**: Yes (for faster training)

**Implementation Details:**
```python
from transformers import Wav2Vec2Model
self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
```

### 2. Transformer Encoder

- **Purpose**: Process Wav2Vec2 features with long-range context
- **Layers**: 4 TransformerEncoderLayers
- **Attention heads**: 8
- **Feedforward dim**: 3072
- **Dropout**: 0.1

**Key Features:**
- Batch-first processing
- Attention mask support
- Layer normalization
- Residual connections

### 3. Detection Head

- **Purpose**: Classify pronunciation correctness
- **Architecture**:
  - Mean pooling with attention mask
  - Linear(768 → 768)
  - Tanh activation
  - Dropout(0.1)
  - Linear(768 → 2)
- **Output**: Logits for [Correct, Mispronounced]
- **Loss**: Cross-entropy

**Prediction Process:**
```python
1. Pool sequence features (mean with mask)
2. Project through FC layers
3. Apply softmax
4. Get class with max probability
```

### 4. Reconstruction Head

- **Purpose**: Generate corrective mel spectrogram
- **Architecture**:
  - Linear projection to mel space
  - CNN postnet for refinement
  - Residual connection
- **Output**: Mel spectrogram (time × 80)
- **Loss**: MSE with target mel

**Processing Pipeline:**
```python
1. Project hidden states to mel bins (768 → 80)
2. Apply CNN postnet (Conv1d layers)
3. Add residual: output = projection + postnet
4. Return mel spectrogram
```

## Training

### Loss Function

```python
Total Loss = α × Detection Loss + β × Reconstruction Loss

Where:
- α (detection_weight) = 0.7 (default)
- β (reconstruction_weight) = 0.3 (default)
- Detection Loss = CrossEntropy(predictions, labels)
- Reconstruction Loss = MSE(predicted_mel, target_mel)
```

### Optimization

**Optimizer:** AdamW
- Learning rate: 1e-5
- Betas: (0.9, 0.999)
- Weight decay: 0.01
- Epsilon: 1e-8

**Scheduler:** Linear warmup + decay
- Warmup steps: 500
- Total steps: num_epochs × steps_per_epoch

**Gradient Management:**
- Gradient clipping: max_norm = 1.0
- Gradient accumulation: 4 steps

### Training Strategy

1. **Stage 1: Frozen Backbone (Optional)**
   - Freeze Wav2Vec2 parameters
   - Train only transformer and heads
   - Faster convergence, lower memory
   - Duration: 10-20 epochs

2. **Stage 2: Full Fine-tuning**
   - Unfreeze all parameters
   - Lower learning rate (1e-5)
   - Better final performance
   - Duration: 30-50 epochs

## Data Processing

### Audio Preprocessing

1. **Loading**: 
   - Read audio file with torchaudio
   - Convert stereo to mono if needed
   - Resample to 16kHz

2. **Normalization**:
   - Standardize waveform amplitude
   - Apply padding/trimming to fixed length

3. **Mel Spectrogram**:
   - FFT size: 1024
   - Hop length: 256
   - Mel bins: 80
   - Log scale: log(mel + 1e-9)

### Dataset Format

**JSON Structure:**
```json
{
  "audio_path": "path/to/audio.wav",
  "label": 0,  // 0=correct, 1=mispronounced
  "text": "transcription",
  "reference_audio": "path/to/reference.wav"  // optional
}
```

**Batch Processing:**
- Automatic padding to max length
- Attention mask generation
- Efficient GPU utilization
- Pin memory for faster transfer

## Inference

### Single File Inference

```python
1. Load audio file
2. Preprocess (resample, pad/trim)
3. Extract features with model
4. Get detection prediction
5. Optionally save reconstruction
```

### Batch Inference

```python
1. Collect audio files from directory
2. Process in batches
3. Aggregate results
4. Save to JSON
```

### Output Format

```python
{
  "prediction": "Mispronounced",
  "confidence": 0.8542,
  "probabilities": {
    "correct": 0.1458,
    "mispronounced": 0.8542
  },
  "reconstructed_mel": numpy_array  // optional
}
```

## Performance Metrics

### Classification Metrics

- **Accuracy**: Correct predictions / Total predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Reconstruction Metrics

- **MSE**: Mean squared error on mel spectrograms
- **MAE**: Mean absolute error
- **Correlation**: Pearson correlation with target

### Expected Performance

With L2-ARCTIC or similar datasets:
- Accuracy: 85-95%
- F1 Score: 0.80-0.92
- Reconstruction MSE: < 0.5

## Memory and Compute

### GPU Memory Usage

| Configuration | Memory | Training Time |
|--------------|--------|---------------|
| Batch size 16, frozen | ~8GB | 4-6 hours |
| Batch size 16, full | ~12GB | 6-8 hours |
| Batch size 32, frozen | ~14GB | 3-4 hours |
| Batch size 32, full | ~20GB | 5-6 hours |

*Note: Times based on 10k samples, 50 epochs*

### Optimization Tips

1. **Reduce Memory**:
   - Lower batch size
   - Freeze Wav2Vec2
   - Use gradient checkpointing
   - Reduce sequence length

2. **Speed Up Training**:
   - Increase batch size
   - Use mixed precision (fp16)
   - Reduce transformer layers
   - Use gradient accumulation

## Code Organization

```
src/
├── models/
│   └── mispronunciation_model.py  # Main model classes
├── data/
│   └── data_loader.py             # Data loading utilities
└── utils/
    └── helpers.py                 # Helper functions

Scripts:
├── train.py      # Training script
├── inference.py  # Inference script
└── examples/
    └── run_examples.py  # Example usage
```

## Extension Points

### Adding New Features

1. **Phoneme-level Detection**:
   - Modify detection head to output per-phoneme scores
   - Use CTC loss for alignment

2. **Multi-class Classification**:
   - Change num_classes in config
   - Update labels in dataset

3. **Different Backbone**:
   - Replace Wav2Vec2 with HuBERT, Wav2Vec2-Large, etc.
   - Update config: `wav2vec2_model: "facebook/hubert-base-ls960"`

4. **Custom Loss Functions**:
   - Implement in model forward method
   - Add loss weights to config

### Dataset Integration

1. **L2-ARCTIC**:
   - Extract audio files
   - Parse annotation files
   - Convert to JSON format

2. **Custom Dataset**:
   - Implement custom Dataset class
   - Override `__getitem__` method
   - Maintain same output format

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch_size
   - Enable gradient_accumulation_steps
   - Freeze Wav2Vec2 backbone

2. **Slow Training**:
   - Increase batch_size
   - Use multiple workers in DataLoader
   - Enable pin_memory=True

3. **Poor Accuracy**:
   - Train longer (more epochs)
   - Check data quality
   - Adjust learning rate
   - Increase model capacity

4. **Import Errors**:
   - Install all requirements
   - Check Python version (3.8+)
   - Verify CUDA compatibility

## References

- Wav2Vec2: https://arxiv.org/abs/2006.11477
- Transformer: https://arxiv.org/abs/1706.03762
- L2-ARCTIC: http://psi.engr.tamu.edu/l2-arctic-corpus/
- PyTorch: https://pytorch.org/
- Hugging Face Transformers: https://huggingface.co/transformers/
