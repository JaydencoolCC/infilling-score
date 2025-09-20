"""
Constants and configuration for infilling score detection.
"""

# Constants for infill calculations
NEXT_TOKEN_LENGTHS = [0, 1, 3, 5, 10, 20, 50, 100, 200, 500]
RATIO_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# model_path_config
model_path_dict = {
    "s1-32B-0.8": "/mnt/sharedata/ssd_large/users/zhanghx/models/reasoner/32B/s1K-32B-0.8/model",
    "s1-32B-0.6": "/mnt/sharedata/ssd_large/users/zhanghx/models/reasoner/32B/s1K-32B-0.6/model",
    "s1.1-32B-0.8": "/mnt/sharedata/ssd_large/users/zhanghx/models/reasoner/32B/s1.1K-32B-0.8/model",
    "limo-32B-0.8": "/mnt/sharedata/ssd_large/users/zhanghx/models/reasoner/32B/limo-32B-0.8/model",
    "s1-14B-0.8": "/mnt/sharedata/ssd_large/users/zhanghx/models/reasoner/different-sized/Qwen2.5-14B-Instruct/s1K/model",
    "s1-7B-0.8": "/mnt/sharedata/ssd_large/users/zhanghx/models/reasoner/different-sized/Qwen2.5-7B-Instruct/s1K/model",
    "limo-14B-0.8":"/mnt/sharedata/hdd/zhanghx/reason/limo_14B",
    "s1.1-14B-0.8":"/mnt/sharedata/hdd/zhanghx/reason/s1.1k_14B",
    "limo-7B-0.8": "/mnt/sharedata/hdd/zhanghx/reason/7B_new/limo_7B",
    "s1.1-7B-0.8": "/mnt/sharedata/hdd/zhanghx/reason/7B_new/s1.1k_7B",
    }


# Available WikiMIA datasets
WIKIMIA_DATASETS = [
    'WikiMIA_length32', 
    'WikiMIA_length64', 
    'WikiMIA_length128',
    'WikiMIA_length32_paraphrased', 
    'WikiMIA_length64_paraphrased', 
    'WikiMIA_length128_paraphrased'
]

# Default configuration
DEFAULT_CONFIG = {
    'model': 'meta-llama/Meta-Llama-3-8B',
    'dataset': 'WikiMIA_length64_paraphrased',
    'batch_size': 16,
    'output_dir': 'results',
    'inf_clip_value': -100.0,
    'score_clip_range': (-50.0, 50.0),
    'ratio_clip_range': (-200.0, 200.0),
    'min_sigma': 1e-6,
}

# Numerical safety settings
NUMERICAL_SAFETY = {
    'inf_replacement': -100.0,
    'min_log_prob': -100.0,
    'max_log_prob': 0.0,
    'min_sigma': 1e-6,
    'score_clip_min': -50.0,
    'score_clip_max': 50.0,
    'ratio_clip_min': -200.0,
    'ratio_clip_max': 200.0,
}

