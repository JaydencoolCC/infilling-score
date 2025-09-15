"""
Infilling Score Detection Package

A modular, optimized implementation for detecting whether text snippets were
part of a language model's training data using various methods including
Min-k, Min-k++, and optimized infill approaches.
"""

from .models.detector import InfillingScoreDetector
from .data.processor import DataProcessor
from .metrics.calculator import MetricsCalculator
from .optimizations.infill import OptimizedInfillCalculator

__version__ = "1.0.0"

__all__ = [
    "InfillingScoreDetector",
    "DataProcessor", 
    "MetricsCalculator",
    "OptimizedInfillCalculator",
]

