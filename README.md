# Infilling Score Detection

A modular, optimized implementation for detecting whether text snippets were part of a language model's training data using WikiMIA datasets.

## 🚀 Features

- **Multiple Detection Methods**: Min-k, Min-k++, infill-based, and basic perplexity methods
- **Optimized Infill**: 4-8x speedup with batched processing
- **Robust Numerical Handling**: Handles BFloat16 precision issues and -inf values
- **Flexible Performance Options**: Half precision, int8 quantization, mixed precision
- **Comprehensive Evaluation**: AUROC, FPR@95%, TPR@5% metrics
- **Modular Design**: Clean, maintainable codebase


## Installation

1. **Clone the repository**:
```bash
git clone <repository_url>
cd infilling_score
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up for development** (optional):
```bash
pip install -e .
```

## Usage

### Basic Usage

```bash
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset WikiMIA_length64_paraphrased \
    --half \
    --clip_inf
```

### Performance Optimizations

**Fastest (with -inf protection):**
```bash
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset WikiMIA_length64_paraphrased \
    --half \
    --clip_inf \
    --batch_size 16
```

**Memory-efficient:**
```bash
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset WikiMIA_length128 \
    --int8 \
    --gradient_checkpointing \
    --batch_size 32
```

**Mixed precision (balanced):**
```bash
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset WikiMIA_length64_paraphrased \
    --mixed_precision \
    --batch_size 16
```

### Available Options

#### Model & Dataset
- `--model`: Model name/path (default: meta-llama/Meta-Llama-3-8B)
- `--dataset`: WikiMIA dataset variant (see options below)

#### Performance Options
- `--half`: Use half precision (BFloat16)
- `--int8`: Use int8 quantization
- `--mixed_precision`: Model in half, calculations in float32
- `--gradient_checkpointing`: Save memory, allow larger batches
- `--batch_size`: Batch size for optimization (default: 16)

#### Numerical Stability
- `--clip_inf`: Clip -inf values (essential with `--half`)
- `--disable_optimized_infill`: Use original slow method

#### Analysis & Debugging
- `--benchmark`: Compare optimized vs original speed
- `--verbose`: Enable detailed output
- `--analyze_scores`: Analyze score distributions

### Available Datasets

- `WikiMIA_length32`: 32-token excerpts
- `WikiMIA_length64`: 64-token excerpts  
- `WikiMIA_length128`: 128-token excerpts
- `WikiMIA_length32_paraphrased`: Paraphrased 32-token
- `WikiMIA_length64_paraphrased`: Paraphrased 64-token (default)
- `WikiMIA_length128_paraphrased`: Paraphrased 128-token

## 📊 Performance Guide

| Configuration | Speed | Memory | Accuracy | Stability |
|---------------|-------|---------|----------|-----------|
| `--half --clip_inf` | 🔥🔥🔥 | 🔥🔥🔥 | ✅ | ✅ |
| `--mixed_precision` | 🔥🔥 | 🔥🔥 | ✅ | ✅ |
| `--int8` | 🔥 | 🔥🔥🔥🔥 | ✅ | ✅ |
| Full precision | 🔥 | 🔥 | ✅ | ✅ |

**Recommended**: `--half --clip_inf` for best speed/stability balance.

## 🔧 Programmatic Usage

```python
from infilling_score import InfillingScoreDetector, DataProcessor

# Initialize detector
detector = InfillingScoreDetector(
    model_name="meta-llama/Meta-Llama-3-8B",
    use_half=True,
    clip_inf=True,
    batch_size=16
)

# Load data
data = DataProcessor.load_wikimia_data("WikiMIA_length64_paraphrased")

# Analyze single text
scores = detector.analyze_text("Your text here")
print(scores)
```

## 🐛 Troubleshooting

### `-inf` Errors
**Problem**: `ValueError: Input contains infinity`
**Solution**: Add `--clip_inf` flag when using `--half`

### Memory Issues
**Solutions**:
- Use `--int8` instead of `--half`
- Add `--gradient_checkpointing`
- Reduce `--batch_size`

### Slow Performance
**Solutions**:
- Add `--half --clip_inf`
- Increase `--batch_size` (if memory allows)
- Ensure optimized infill is enabled (default)

## Results

Results are saved as CSV files in the `results/` directory with format:
`{model_name}_{dataset}_{config}_results.csv`

Key metrics:
- **AUROC**: Area under ROC curve (higher = better)
- **FPR@95%**: False positive rate at 95% true positive rate (lower = better)  
- **TPR@5%**: True positive rate at 5% false positive rate (higher = better)

