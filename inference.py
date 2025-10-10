"""
Inference script for mispronunciation detection and speech reconstruction.
"""

import torch
import argparse
import os
import sys
import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.mispronunciation_model import MispronunciationDetectionModel
from src.data.data_loader import AudioProcessor
from src.utils.helpers import load_config, load_checkpoint


class MispronunciationPredictor:
    """Inference class for mispronunciation detection."""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
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
        
        # Create model
        self.model = MispronunciationDetectionModel(model_config)
        
        # Load checkpoint
        load_checkpoint(model_path, self.model)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=config['data']['sample_rate'],
            max_length=int(config['data']['max_audio_length'] * config['data']['sample_rate'])
        )
        
        self.labels = ['Correct', 'Mispronounced']
        
    @torch.no_grad()
    def predict(self, audio_path: str, return_reconstruction: bool = False):
        """
        Predict mispronunciation for an audio file.
        
        Args:
            audio_path: Path to audio file
            return_reconstruction: Whether to return reconstructed mel spectrogram
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess audio
        waveform = self.audio_processor.load_audio(audio_path)
        actual_length = min(len(waveform), self.audio_processor.max_length)
        waveform = self.audio_processor.pad_or_trim(waveform)
        
        # Create attention mask
        attention_mask = torch.zeros(len(waveform))
        attention_mask[:actual_length] = 1
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_values=waveform,
            attention_mask=attention_mask
        )
        
        # Get prediction
        logits = outputs['detection_logits']
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
        
        result = {
            'prediction': self.labels[pred_class],
            'confidence': confidence,
            'probabilities': {
                'correct': probs[0, 0].item(),
                'mispronounced': probs[0, 1].item()
            }
        }
        
        if return_reconstruction:
            result['reconstructed_mel'] = outputs['reconstruction_output'].cpu().numpy()
        
        return result
    
    @torch.no_grad()
    def batch_predict(self, audio_paths: list):
        """Predict for multiple audio files."""
        results = []
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            result['audio_path'] = audio_path
            results.append(result)
        return results


def main(args):
    """Main inference function."""
    # Create predictor
    print("Loading model...")
    predictor = MispronunciationPredictor(
        args.model_path,
        args.config,
        device=args.device
    )
    
    if args.audio_file:
        # Single file prediction
        print(f"\nAnalyzing: {args.audio_file}")
        result = predictor.predict(args.audio_file, return_reconstruction=args.save_reconstruction)
        
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nProbabilities:")
        print(f"  Correct: {result['probabilities']['correct']:.4f}")
        print(f"  Mispronounced: {result['probabilities']['mispronounced']:.4f}")
        
        # Save reconstruction if requested
        if args.save_reconstruction and 'reconstructed_mel' in result:
            output_path = args.output_path or 'reconstructed_mel.npy'
            np.save(output_path, result['reconstructed_mel'])
            print(f"\nReconstructed mel spectrogram saved to: {output_path}")
    
    elif args.audio_dir:
        # Batch prediction
        print(f"\nAnalyzing directory: {args.audio_dir}")
        audio_files = [
            os.path.join(args.audio_dir, f)
            for f in os.listdir(args.audio_dir)
            if f.endswith(('.wav', '.mp3', '.flac'))
        ]
        
        if not audio_files:
            print("No audio files found in directory.")
            return
        
        print(f"Found {len(audio_files)} audio files.")
        results = predictor.batch_predict(audio_files)
        
        # Print results
        print("\nResults:")
        print("-" * 80)
        for result in results:
            print(f"{os.path.basename(result['audio_path'])}: "
                  f"{result['prediction']} (confidence: {result['confidence']:.4f})")
        
        # Save to file if requested
        if args.output_path:
            import json
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output_path}")
    
    else:
        print("Please provide either --audio_file or --audio_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for mispronunciation detection')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--audio_file', type=str, default='',
                        help='Path to single audio file')
    parser.add_argument('--audio_dir', type=str, default='',
                        help='Path to directory with audio files')
    parser.add_argument('--output_path', type=str, default='',
                        help='Path to save results')
    parser.add_argument('--save_reconstruction', action='store_true',
                        help='Save reconstructed mel spectrogram')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    
    args = parser.parse_args()
    main(args)
