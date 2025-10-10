# Mispronunciation Detection Using Speech Reconstruction

A deep learning model that jointly trains mispronunciation detection and speech reconstruction using Wav2Vec2.0 + Transformer layers. This system improves pronunciation feedback by generating corrective speech features, helping language learners identify and correct pronunciation errors.

## 🎯 Overview

This project implements a dual-task model that:
1. **Detects Mispronunciations**: Classifies speech as correctly or incorrectly pronounced
2. **Reconstructs Speech**: Generates mel spectrogram features representing the correct pronunciation

The model uses a Wav2Vec2 backbone for feature extraction, followed by Transformer encoder layers and dual task-specific heads for detection and reconstruction.

## 🏗️ Architecture

```
Input Audio (Waveform)
         ↓
  Wav2Vec2 Feature Extraction
         ↓
  Transformer Encoder Layers
         ↓
    ┌────────┴────────┐
    ↓                 ↓
Detection Head    Reconstruction Head
(Classification)  (Mel Spectrogram)
    ↓                 ↓
Correct/Incorrect  Corrected Speech Features
```

### Key Components

- **Wav2Vec2 Backbone**: Pre-trained speech feature extractor from Facebook AI
- **Transformer Encoder**: 4 layers of multi-head self-attention for contextual processing
- **Detection Head**: Binary classification for mispronunciation detection
- **Reconstruction Head**: Generates mel spectrogram features for speech reconstruction
- **Joint Training**: Combined loss function balancing both tasks

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers (Hugging Face)
- torchaudio
- librosa
- Other dependencies in `requirements.txt`

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/shanks-not-found/imporving-mispronunciation-detection-using-speech-reconstruction.git
cd imporving-mispronunciation-detection-using-speech-reconstruction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

This model works with public Computer-Assisted Pronunciation Training (CAPT) datasets. Recommended datasets:

- **L2-ARCTIC**: Non-native English speech corpus
- **speechocean762**: Pronunciation assessment dataset
- **EpaDB**: English Pronunciation Assessment Database

### Dataset Format

Prepare your data in JSON format:

```json
[
  {
    "audio_path": "path/to/audio1.wav",
    "label": 0,
    "text": "the quick brown fox",
    "reference_audio": "path/to/reference1.wav"
  },
  {
    "audio_path": "path/to/audio2.wav",
    "label": 1,
    "text": "jumps over the lazy dog"
  }
]
```

Where:
- `label`: 0 = correct pronunciation, 1 = mispronounced
- `audio_path`: Path to the audio file
- `text`: Transcription (optional)
- `reference_audio`: Path to correct pronunciation reference (optional)

## 🎓 Training

### Basic Training

```bash
python train.py \
    --config configs/default_config.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --log_dir logs/experiment1
```

### Advanced Options

```bash
python train.py \
    --config configs/default_config.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --test_data data/test.json \
    --freeze_wav2vec2 \
    --log_dir logs/experiment1 \
    --seed 42
```

### Resume Training

```bash
python train.py \
    --config configs/default_config.yaml \
    --train_data data/train.json \
    --val_data data/val.json \
    --resume checkpoints/checkpoint_epoch_10.pt
```

### Training Configuration

Edit `configs/default_config.yaml` to customize:
- Model architecture (hidden size, layers, attention heads)
- Training hyperparameters (learning rate, batch size, epochs)
- Data settings (sample rate, max audio length)
- Loss weights (detection vs reconstruction)

## 🔍 Inference

### Single File Prediction

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --audio_file test_audio.wav
```

### Batch Prediction

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --audio_dir path/to/audio/directory \
    --output_path results.json
```

### Save Reconstructed Speech

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --audio_file test_audio.wav \
    --save_reconstruction \
    --output_path reconstructed.npy
```

## 📝 Examples

Run the examples to understand the model:

```bash
python examples/run_examples.py
```

This will demonstrate:
1. Model forward pass
2. Data loading and preprocessing
3. Training pipeline
4. Inference pipeline

## 🔧 Model Configuration

Key parameters in `configs/default_config.yaml`:

```yaml
model:
  wav2vec2_model: "facebook/wav2vec2-base"
  hidden_size: 768
  num_transformer_layers: 4
  num_attention_heads: 8
  num_pronunciation_classes: 2

training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.00001
  
reconstruction:
  mel_bins: 80
  detection_weight: 0.7
  reconstruction_weight: 0.3
```

## 📂 Project Structure

```
.
├── configs/
│   └── default_config.yaml      # Model and training configuration
├── src/
│   ├── models/
│   │   └── mispronunciation_model.py  # Main model architecture
│   ├── data/
│   │   └── data_loader.py       # Data loading and preprocessing
│   └── utils/
│       └── helpers.py           # Utility functions
├── examples/
│   └── run_examples.py          # Example usage scripts
├── train.py                     # Training script
├── inference.py                 # Inference script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎯 Use Cases

1. **Language Learning Apps**: Provide real-time pronunciation feedback
2. **Speech Therapy**: Assist in pronunciation assessment and correction
3. **Accent Reduction**: Help learners improve pronunciation accuracy
4. **CAPT Systems**: Enhance computer-assisted pronunciation training
5. **Educational Platforms**: Integrate into online language learning platforms

## 🔬 Technical Details

### Model Components

1. **Wav2Vec2 Feature Extractor**
   - Pre-trained on large speech corpus
   - Extracts rich acoustic features
   - Can be fine-tuned or frozen

2. **Transformer Encoder**
   - Multi-head self-attention
   - Captures long-range dependencies
   - Contextual feature processing

3. **Detection Head**
   - Binary classification
   - Mean pooling with attention mask
   - Outputs pronunciation correctness

4. **Reconstruction Head**
   - Mel spectrogram generation
   - Postnet refinement
   - Generates corrective speech features

### Loss Function

```
Total Loss = α × Detection Loss + β × Reconstruction Loss
```

Where:
- Detection Loss: Cross-entropy for classification
- Reconstruction Loss: MSE between predicted and target mel spectrograms
- α (detection_weight): 0.7 (default)
- β (reconstruction_weight): 0.3 (default)

## 📊 Evaluation Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Reconstruction Loss**: MSE of mel spectrograms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Wav2Vec2**: Facebook AI Research
- **Transformers**: Hugging Face
- **CAPT Datasets**: L2-ARCTIC, speechocean762, EpaDB communities

## 📚 References

- [Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [L2-ARCTIC: A Non-native English Speech Corpus](http://psi.engr.tamu.edu/l2-arctic-corpus/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a research/educational implementation. For production use, additional optimizations and testing may be required.