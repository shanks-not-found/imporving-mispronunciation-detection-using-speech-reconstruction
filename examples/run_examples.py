"""
Example script demonstrating how to use the mispronunciation detection model.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.mispronunciation_model import MispronunciationDetectionModel
from src.data.data_loader import AudioProcessor
from src.utils.helpers import load_config


def example_model_forward():
    """Example of creating and running forward pass through the model."""
    
    print("=" * 80)
    print("Example: Model Forward Pass")
    print("=" * 80)
    
    # Load config
    config_path = 'configs/default_config.yaml'
    config = load_config(config_path)
    
    # Setup model config
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
    
    print("\n1. Creating model...")
    model = MispronunciationDetectionModel(model_config)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\n2. Creating dummy input...")
    batch_size = 2
    seq_length = 16000 * 5  # 5 seconds at 16kHz
    
    # Create dummy input
    input_values = torch.randn(batch_size, seq_length)
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.tensor([0, 1])  # 0: correct, 1: mispronounced
    target_mel = torch.randn(batch_size, config['reconstruction']['target_length'], 
                            config['reconstruction']['mel_bins'])
    
    print(f"   Input shape: {input_values.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")
    print(f"   Labels: {labels}")
    print(f"   Target mel shape: {target_mel.shape}")
    
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            target_mel=target_mel
        )
    
    print(f"   Detection logits shape: {outputs['detection_logits'].shape}")
    print(f"   Reconstruction output shape: {outputs['reconstruction_output'].shape}")
    print(f"   Total loss: {outputs['loss'].item():.4f}")
    print(f"   Detection loss: {outputs['detection_loss'].item():.4f}")
    print(f"   Reconstruction loss: {outputs['reconstruction_loss'].item():.4f}")
    
    # Get predictions
    probs = torch.softmax(outputs['detection_logits'], dim=-1)
    preds = torch.argmax(probs, dim=-1)
    print(f"\n   Predictions: {preds}")
    print(f"   Probabilities: {probs}")
    
    print("\n✓ Example completed successfully!")


def example_data_loading():
    """Example of loading and preprocessing audio data."""
    
    print("\n" + "=" * 80)
    print("Example: Data Loading and Preprocessing")
    print("=" * 80)
    
    # Load config
    config_path = 'configs/default_config.yaml'
    config = load_config(config_path)
    
    print("\n1. Creating AudioProcessor...")
    audio_processor = AudioProcessor(
        sample_rate=config['data']['sample_rate'],
        max_length=int(config['data']['max_audio_length'] * config['data']['sample_rate'])
    )
    print(f"   Sample rate: {audio_processor.sample_rate} Hz")
    print(f"   Max length: {audio_processor.max_length} samples "
          f"({audio_processor.max_length / audio_processor.sample_rate:.1f} seconds)")
    
    print("\n2. Creating synthetic audio...")
    import numpy as np
    # Generate a simple sine wave
    duration = 3.0  # seconds
    frequency = 440.0  # Hz (A4 note)
    t = np.linspace(0, duration, int(audio_processor.sample_rate * duration))
    waveform = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    waveform_tensor = torch.from_numpy(waveform)
    
    print(f"   Generated waveform shape: {waveform_tensor.shape}")
    print(f"   Duration: {len(waveform_tensor) / audio_processor.sample_rate:.2f} seconds")
    
    print("\n3. Padding/trimming waveform...")
    processed_waveform = audio_processor.pad_or_trim(waveform_tensor)
    print(f"   Processed waveform shape: {processed_waveform.shape}")
    
    print("\n4. Extracting mel spectrogram...")
    mel_spec = audio_processor.extract_mel_spectrogram(
        processed_waveform,
        n_mels=config['reconstruction']['mel_bins']
    )
    print(f"   Mel spectrogram shape: {mel_spec.shape}")
    print(f"   Mel bins: {mel_spec.shape[1]}")
    print(f"   Time steps: {mel_spec.shape[0]}")
    
    print("\n✓ Example completed successfully!")


def example_inference():
    """Example of running inference on audio."""
    
    print("\n" + "=" * 80)
    print("Example: Inference Pipeline")
    print("=" * 80)
    
    print("\n1. This example shows how to use the trained model for inference.")
    print("   After training, you can run:")
    print()
    print("   python inference.py \\")
    print("       --model_path checkpoints/best_model.pt \\")
    print("       --config configs/default_config.yaml \\")
    print("       --audio_file path/to/audio.wav")
    print()
    print("2. For batch inference:")
    print()
    print("   python inference.py \\")
    print("       --model_path checkpoints/best_model.pt \\")
    print("       --config configs/default_config.yaml \\")
    print("       --audio_dir path/to/audio/directory \\")
    print("       --output_path results.json")
    print()
    print("3. To save reconstructed speech features:")
    print()
    print("   python inference.py \\")
    print("       --model_path checkpoints/best_model.pt \\")
    print("       --config configs/default_config.yaml \\")
    print("       --audio_file path/to/audio.wav \\")
    print("       --save_reconstruction \\")
    print("       --output_path reconstructed.npy")


def example_training():
    """Example of training workflow."""
    
    print("\n" + "=" * 80)
    print("Example: Training Pipeline")
    print("=" * 80)
    
    print("\n1. Prepare your dataset in JSON format:")
    print("   [")
    print("     {")
    print('       "audio_path": "path/to/audio1.wav",')
    print('       "label": 0,  # 0 for correct, 1 for mispronounced')
    print('       "text": "the quick brown fox"')
    print("     },")
    print("     ...")
    print("   ]")
    
    print("\n2. Start training:")
    print()
    print("   python train.py \\")
    print("       --config configs/default_config.yaml \\")
    print("       --train_data data/train.json \\")
    print("       --val_data data/val.json \\")
    print("       --log_dir logs/experiment1")
    
    print("\n3. Resume from checkpoint:")
    print()
    print("   python train.py \\")
    print("       --config configs/default_config.yaml \\")
    print("       --train_data data/train.json \\")
    print("       --val_data data/val.json \\")
    print("       --resume checkpoints/checkpoint_epoch_10.pt")
    
    print("\n4. Fine-tune with frozen Wav2Vec2:")
    print()
    print("   python train.py \\")
    print("       --config configs/default_config.yaml \\")
    print("       --train_data data/train.json \\")
    print("       --val_data data/val.json \\")
    print("       --freeze_wav2vec2")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("MISPRONUNCIATION DETECTION EXAMPLES")
    print("=" * 80)
    
    try:
        # Run examples
        example_model_forward()
        example_data_loading()
        example_training()
        example_inference()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Prepare your dataset using public CAPT datasets (L2-ARCTIC, speechocean762, etc.)")
        print("2. Update configs/default_config.yaml with your settings")
        print("3. Run train.py to train the model")
        print("4. Use inference.py to evaluate on new audio")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
