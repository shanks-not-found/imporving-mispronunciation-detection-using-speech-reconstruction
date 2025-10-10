# Quick Start Guide

This guide will help you get started with the Mispronunciation Detection system.

## Setup

### 1. Install Dependencies

```bash
# Run the setup script (recommended)
bash setup.sh

# Or manually:
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python examples/run_examples.py
```

This will run demonstrations of the model without requiring a dataset.

## Prepare Your Dataset

### Option A: Use Public CAPT Datasets

Download one of these public datasets:

1. **L2-ARCTIC** (Recommended for beginners)
   - Website: http://psi.engr.tamu.edu/l2-arctic-corpus/
   - Contains non-native English speech with annotations
   
2. **speechocean762**
   - Available on GitHub
   - Pronunciation assessment dataset
   
3. **EpaDB**
   - English Pronunciation Assessment Database

### Option B: Prepare Your Own Dataset

Create a JSON file with your audio annotations:

```json
[
  {
    "audio_path": "data/audio/sample1.wav",
    "label": 0,
    "text": "hello world"
  },
  {
    "audio_path": "data/audio/sample2.wav",
    "label": 1,
    "text": "pronunciation error"
  }
]
```

**Labels:**
- `0` = Correct pronunciation
- `1` = Mispronounced

## Training

### Basic Training

```bash
python train.py \
    --config configs/default_config.yaml \
    --train_data data/train.json \
    --val_data data/val.json
```

### Monitor Training

```bash
# In a separate terminal
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### Training Tips

1. **Start with frozen Wav2Vec2** (faster training):
   ```bash
   python train.py --freeze_wav2vec2 --train_data data/train.json --val_data data/val.json
   ```

2. **Fine-tune the full model** (better performance):
   ```bash
   python train.py --train_data data/train.json --val_data data/val.json
   ```

3. **Adjust hyperparameters** in `configs/default_config.yaml`:
   - Reduce `batch_size` if you have memory issues
   - Increase `num_epochs` for better convergence
   - Adjust `learning_rate` for faster/slower learning

## Inference

### Single Audio File

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --audio_file test.wav
```

**Output:**
```
Prediction: Mispronounced
Confidence: 0.8542

Probabilities:
  Correct: 0.1458
  Mispronounced: 0.8542
```

### Batch Processing

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --audio_dir data/test_audio/ \
    --output_path results.json
```

### Get Reconstructed Speech Features

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --audio_file test.wav \
    --save_reconstruction \
    --output_path reconstructed.npy
```

## Configuration

Edit `configs/default_config.yaml` to customize the model:

### Key Settings

```yaml
# Model architecture
model:
  num_transformer_layers: 4    # More layers = more capacity
  hidden_size: 768             # Larger = more capacity
  
# Training
training:
  batch_size: 16               # Reduce if GPU memory issues
  learning_rate: 0.00001       # Lower = more stable
  num_epochs: 50               # More = better convergence
  
# Task balance
reconstruction:
  detection_weight: 0.7        # Focus on detection
  reconstruction_weight: 0.3   # Focus on reconstruction
```

## Troubleshooting

### Out of Memory

1. Reduce `batch_size` in config
2. Use `--freeze_wav2vec2` flag
3. Reduce `max_audio_length` in config

### Low Accuracy

1. Train for more epochs
2. Check data quality and labels
3. Adjust learning rate
4. Increase model capacity (more layers/hidden size)

### Slow Training

1. Use `--freeze_wav2vec2` initially
2. Reduce `gradient_accumulation_steps`
3. Use a GPU if available
4. Reduce `num_transformer_layers`

## Examples

### Example 1: Quick Test with Frozen Backbone

```bash
python train.py \
    --config configs/default_config.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --freeze_wav2vec2 \
    --log_dir logs/quick_test
```

### Example 2: Full Training with Checkpointing

```bash
python train.py \
    --config configs/default_config.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --test_data data/test.json \
    --log_dir logs/full_training \
    --save_every 5
```

### Example 3: Resume from Checkpoint

```bash
python train.py \
    --config configs/default_config.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --resume checkpoints/checkpoint_epoch_10.pt
```

## Expected Results

With proper training on a good dataset:

- **Accuracy**: 85-95%
- **F1 Score**: 0.80-0.92
- **Training time**: 4-8 hours on GPU (depending on dataset size)

## Next Steps

1. ✅ Install and verify setup
2. ✅ Run examples
3. ✅ Prepare/download dataset
4. ✅ Start training
5. ✅ Evaluate on test set
6. ✅ Use for inference

## Resources

- **README.md**: Comprehensive documentation
- **configs/**: Configuration files
- **examples/**: Example scripts
- **Model architecture**: See `src/models/mispronunciation_model.py`

## Support

For issues or questions:
1. Check the README.md
2. Review example scripts
3. Open an issue on GitHub

---

Happy training! 🚀
