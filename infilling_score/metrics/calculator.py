"""
Metrics calculation for infilling score evaluation.
"""

from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc


class MetricsCalculator:
    """Calculate evaluation metrics with robust error handling."""
    
    @staticmethod
    def calculate_metrics(scores: List[float], labels: List[int], verbose: bool = False) -> Tuple[float, float, float]:
        """
        Calculate AUROC, FPR@95%, and TPR@5% with robust error handling.
        
        Args:
            scores: List of prediction scores
            labels: List of ground truth labels (1 for training, 0 for non-training)
            verbose: Whether to print debugging information
            
        Returns:
            Tuple of (auroc, fpr95, tpr05, tpr01)
        """
        # Convert to numpy arrays for easier handling
        scores_array = np.array(scores, dtype=np.float64)
        labels_array = np.array(labels)
        
        # Check for invalid values
        finite_mask = np.isfinite(scores_array)
        inf_count = np.sum(np.isinf(scores_array))
        nan_count = np.sum(np.isnan(scores_array))
        finite_count = np.sum(finite_mask)
        
        if verbose:
            print(f"Score analysis: Total={len(scores_array)}, Finite={finite_count}, Inf={inf_count}, NaN={nan_count}")
        
        if inf_count > 0 and verbose:
            print(f"Found {inf_count} infinite values in scores")
        
        if nan_count > 0 and verbose:
            print(f"Found {nan_count} NaN values in scores")
        
        # Filter out all non-finite values (inf, -inf, nan)
        if not np.all(finite_mask):
            invalid_count = len(scores_array) - finite_count
            if verbose:
                print(f"Filtering out {invalid_count} invalid scores")
            scores_array = scores_array[finite_mask]
            labels_array = labels_array[finite_mask]
        
        # Check if we have enough valid data after filtering
        if len(scores_array) < 2:
            if verbose:
                print("Warning: Not enough valid scores for metrics calculation")
            return 0.0, 1.0, 0.0, 0.0
        
        # Check if we have both classes after filtering
        unique_labels = np.unique(labels_array)
        if len(unique_labels) < 2:
            if verbose:
                print("Warning: Only one class present in labels after filtering")
            return 0.0, 1.0, 0.0, 0.0
        
        # Final check: ensure no extreme values that could cause issues
        # Replace any remaining extreme values with clipped versions
        scores_array = np.clip(scores_array, -1e10, 1e10)
        
        try:
            fpr_list, tpr_list, _ = roc_curve(labels_array, scores_array)
            auroc = auc(fpr_list, tpr_list)
            
            # FPR when TPR >= 95%
            fpr95_idx = np.where(tpr_list >= 0.95)[0]
            fpr95 = fpr_list[fpr95_idx[0]] if len(fpr95_idx) > 0 else 1.0
            
            # TPR when FPR <= 5%
            tpr05_idx = np.where(fpr_list <= 0.05)[0]
            tpr05 = tpr_list[tpr05_idx[-1]] if len(tpr05_idx) > 0 else 0.0
            
            # TPR when FPR <= 1%
            tpr01_idx = np.where(fpr_list <= 0.01)[0]
            tpr01 = tpr_list[tpr01_idx[-1]] if len(tpr01_idx) > 0 else 0.0
            return auroc, fpr95, tpr05, tpr01
            
        except Exception as e:
            if verbose:
                print(f"Error in ROC calculation: {e}")
                print(f"Scores range: [{np.min(scores_array):.3f}, {np.max(scores_array):.3f}]")
                print(f"Labels: {np.unique(labels_array, return_counts=True)}")
            return 0.0, 1.0, 0.0, 0.0
    
    @staticmethod
    def analyze_score_distribution(scores: List[float], method_name: str = "") -> None:
        """Analyze and print score distribution statistics."""
        scores_array = np.array(scores, dtype=np.float64)
        
        finite_count = np.sum(np.isfinite(scores_array))
        inf_count = np.sum(np.isinf(scores_array))
        nan_count = np.sum(np.isnan(scores_array))
        total_count = len(scores_array)
        
        print(f"Score analysis for {method_name}:")
        print(f"  Total: {total_count}")
        print(f"  Finite: {finite_count} ({finite_count/total_count:.1%})")
        print(f"  Infinite: {inf_count} ({inf_count/total_count:.1%})")
        print(f"  NaN: {nan_count} ({nan_count/total_count:.1%})")
        
        if finite_count > 0:
            finite_scores = scores_array[np.isfinite(scores_array)]
            print(f"  Finite range: [{np.min(finite_scores):.3f}, {np.max(finite_scores):.3f}]")
            print(f"  Finite mean: {np.mean(finite_scores):.3f}")
            print(f"  Finite std: {np.std(finite_scores):.3f}")

