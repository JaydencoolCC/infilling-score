#!/usr/bin/env python3
"""
Main execution script for infilling score detection.

Usage:
    python main.py --model EleutherAI/pythia-6.9b  --dataset WikiMIA_length64_paraphrased --half --clip_inf
"""

import argparse
import time
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from infilling_score.models.detector import InfillingScoreDetector
from infilling_score.data.processor import DataProcessor  
from infilling_score.metrics.calculator import MetricsCalculator
from infilling_score.optimizations.infill import OptimizedInfillCalculator
from infilling_score.utils.constants import WIKIMIA_DATASETS, DEFAULT_CONFIG, model_path_dict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimized Infilling Score Detection')
    
    # Model and dataset options
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model'],
                       help='Model name or path')
    parser.add_argument('--dataset', type=str, default=DEFAULT_CONFIG['dataset'],
                       help='WikiMIA dataset to use')
    
    # Performance options
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                       help='Batch size for optimized infill calculation')
    parser.add_argument('--half', action='store_true', help='Use half precision')
    parser.add_argument('--int8', action='store_true', help='Use int8 quantization')
    parser.add_argument('--mixed_precision', action='store_true', 
                       help='Use mixed precision: half for model, float32 for calculations')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory (allows larger batch sizes)')
    
    # Optimization options
    parser.add_argument('--disable_optimized_infill', action='store_true',
                       help='Disable optimized infill (use original slower method)')
    parser.add_argument('--clip_inf', action='store_true',
                       help='Clip -inf values to finite numbers (allows --half without errors)')
    
    # Output and debugging options
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                       help='Output directory for results')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparing optimized vs original icnfill')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--analyze_scores', action='store_true',
                       help='Analyze score distributions for debugging')
    
    return parser.parse_args()


def run_benchmark(detector: InfillingScoreDetector, test_sample: str, args) -> None:
    """Run benchmark comparing optimized vs original infill methods."""
    print("\nRunning benchmark...")
    
    print("Testing optimized infill...")
    start_time = time.time()
    detector.use_optimized_infill = True
    detector.infill_calculator = OptimizedInfillCalculator(
        detector.model, detector.tokenizer, detector.device, args.batch_size, args.clip_inf
    )
    optimized_scores = detector.calculate_infill_scores(test_sample)
    optimized_time = time.time() - start_time
    
    print("Testing original infill...")
    start_time = time.time()
    detector.use_optimized_infill = False
    original_scores = detector.calculate_infill_scores(test_sample)
    original_time = time.time() - start_time
    
    print(f"\nBenchmark Results:")
    print(f"Original method: {original_time:.2f} seconds")
    print(f"Optimized method: {optimized_time:.2f} seconds")
    print(f"Speedup: {original_time/optimized_time:.1f}x")
    
    # Check if results match (within tolerance)
    if len(optimized_scores[0]) > 0 and len(original_scores[0]) > 0:
        results_match = np.allclose(list(optimized_scores[0]), list(original_scores[0]), rtol=1e-5)
        print(f"Results match: {results_match}")
    else:
        print("Cannot compare results (empty score arrays)")


def analyze_score_distributions(all_scores: Dict[str, List[float]], verbose: bool = False) -> None:
    """Analyze and print score distribution statistics."""
    print("\nScore Distribution Analysis:")
    print("=" * 50)
    
    methods_with_issues = []
    
    for method, scores in all_scores.items():
        if verbose or any(not np.isfinite(score) for score in scores):
            print(f"\nMethod: {method}")
            MetricsCalculator.analyze_score_distribution(scores, method)
            
            # Count issues
            scores_array = np.array(scores)
            finite_count = np.sum(np.isfinite(scores_array))
            if finite_count != len(scores_array):
                methods_with_issues.append(method)
    
    if methods_with_issues:
        print(f"\nMethods with numerical issues: {len(methods_with_issues)}")
        for method in methods_with_issues[:5]:  # Show first 5
            print(f"  - {method}")
        if len(methods_with_issues) > 5:
            print(f"  ... and {len(methods_with_issues) - 5} more")
    else:
        print("\nAll methods have finite scores ✓")


def save_results(results: pd.DataFrame, args, model_id: str, dataset_id: str) -> str:
    """Save results to CSV file."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create descriptive filename
    suffix = "_optimized" if not args.disable_optimized_infill else "_original"
    if args.half and args.clip_inf:
        suffix += "_half_clipped"
    elif args.half:
        suffix += "_half"
    elif args.int8:
        suffix += "_int8"
    elif args.mixed_precision:
        suffix += "_mixed"
    
    output_file = os.path.join(args.output_dir, f"{model_id}_{dataset_id}{suffix}_results.csv")
    results.to_csv(output_file, index=False)
    
    return output_file


def display_top_methods(results: pd.DataFrame, top_k: int = 5) -> None:
    """Display top performing methods."""
    print(f"\nTop {top_k} performing methods by AUROC:")
    print("=" * 50)
    
    df_sorted = results.copy()
    df_sorted['auroc_numeric'] = df_sorted['auroc'].str.rstrip('%').astype(float)
    top_methods = df_sorted.nlargest(top_k, 'auroc_numeric')[['method', 'auroc', 'fpr95', 'tpr05']]
    print(top_methods.to_string(index=False))
    
    return top_methods
    


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("Infilling Score Detection")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    
    
    # get_model_path
    args.model = model_path_dict[args.model]
    
    # Initialize detector
    use_optimized = not args.disable_optimized_infill
    detector = InfillingScoreDetector(
        args.model, args.half, args.int8, args.batch_size, use_optimized,
        args.mixed_precision, args.clip_inf, args.gradient_checkpointing
    )
    
    # Load WikiMIA data
    # print(f"\nLoading WikiMIA dataset: {args.dataset}")
    # data = DataProcessor.load_wikimia_data(args.dataset)
    # Print dataset statistics
    # DataProcessor.print_dataset_statistics(data)
    
    # load reasoning data
    # print(f"\nLoading reasoning dataset: {args.dataset}")
    data = DataProcessor.load_reasoning_data(args.dataset)
    # data = data[:5]
    DataProcessor.print_dataset_statistics(data)
    # import pdb; pdb.set_trace()
    
    # Run benchmark if requested
    if args.benchmark and len(data) > 0:
        run_benchmark(detector, data[0]['input'], args)
        
        # Restore optimized setting
        detector.use_optimized_infill = use_optimized
        if use_optimized:
            detector.infill_calculator = OptimizedInfillCalculator(
                detector.model, detector.tokenizer, detector.device, args.batch_size, args.clip_inf
            )
    
    # Run inference
    print("\nRunning membership inference detection...")
    all_scores = defaultdict(list)
    
    for sample in tqdm(data, desc='Processing samples'):
        text = sample['input']
        scores = detector.analyze_text(text)
        
        for method, score in scores.items():
            all_scores[method].append(score)
            
    
    # Analyze score distributions if requested
    if args.analyze_scores:
        analyze_score_distributions(all_scores, args.verbose)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    labels = [sample['label'] for sample in data]
    results = defaultdict(list)
    
    for method, scores in all_scores.items():
        auroc, fpr95, tpr05, tpr01= MetricsCalculator.calculate_metrics(scores, labels, args.verbose)
        
        results['method'].append(method)
        results['auroc'].append(f"{auroc:.1%}")
        results['fpr95'].append(f"{fpr95:.1%}")
        results['tpr05'].append(f"{tpr05:.1%}")
        results['tpr01'].append(f"{tpr01:.1%}")
        
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Display results
    print("\nAll Results:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Display top methods
    top_methods = display_top_methods(df)
        # Save results
    df = pd.concat([df, top_methods], ignore_index=True)
    
    # get best results for filling
    infill_methods = df[df['method'].str.contains('infill')]
    # 找到 auroc 最大的条目
    infill_methods['auroc_numeric'] = infill_methods['auroc'].str.rstrip('%').astype(float)  # 将 auroc 转换为数值
    max_infill_row = infill_methods.loc[infill_methods['auroc_numeric'].idxmax()]  # 找到 auroc 最大的行

    # 将最大值条目添加到 df 中
    max_infill_row = max_infill_row.to_frame().T  # 转换为 DataFrame
    max_infill_row['is_max_infill'] = True  # 添加标记列
    df['is_max_infill'] = False  # 为原始数据添加标记列
    df = pd.concat([df, max_infill_row], ignore_index=True)  # 合并到 df 中
    # 删除临时列
    df.drop(columns=['auroc_numeric'], inplace=True)

    
    model_id = args.model.split('/')[-1]
    dataset_id = args.dataset
    output_file = save_results(df, args, model_id, dataset_id)
    print(f"\nResults saved to: {output_file}")
if __name__ == "__main__":
    main()

