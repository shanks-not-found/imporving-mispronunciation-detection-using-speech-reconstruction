"""Utility functions module."""
from .helpers import (
    load_config,
    save_checkpoint,
    load_checkpoint,
    compute_metrics,
    AverageMeter,
    get_lr,
    format_time,
    count_parameters,
    set_seed
)

try:
    from .visualization import (
        plot_waveform,
        plot_mel_spectrogram,
        plot_comparison,
        plot_prediction_confidence,
        plot_attention_weights,
        plot_training_history,
        visualize_batch_predictions
    )
    _has_visualization = True
except ImportError:
    _has_visualization = False

__all__ = [
    'load_config',
    'save_checkpoint',
    'load_checkpoint',
    'compute_metrics',
    'AverageMeter',
    'get_lr',
    'format_time',
    'count_parameters',
    'set_seed'
]

if _has_visualization:
    __all__.extend([
        'plot_waveform',
        'plot_mel_spectrogram',
        'plot_comparison',
        'plot_prediction_confidence',
        'plot_attention_weights',
        'plot_training_history',
        'visualize_batch_predictions'
    ])
