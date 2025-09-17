"""
Main membership inference detector class.
"""

import copy
import zlib
from typing import Dict, List, Any
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..optimizations.infill import OptimizedInfillCalculator
from ..utils.constants import RATIO_THRESHOLDS, NEXT_TOKEN_LENGTHS, NUMERICAL_SAFETY


class InfillingScoreDetector:
    """Main class for running infilling scores with optimized infill."""
    
    def __init__(self, model_name: str, use_half: bool = False, use_int8: bool = False, 
                 batch_size: int = 16, use_optimized_infill: bool = True, 
                 mixed_precision: bool = False, clip_inf: bool = False,
                 gradient_checkpointing: bool = False):
        """Initialize the detector with a specified model."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_optimized_infill = use_optimized_infill
        self.mixed_precision = mixed_precision
        self.clip_inf = clip_inf
        self.model, self.tokenizer = self._load_model(use_half, use_int8, mixed_precision, gradient_checkpointing)
        self.device = self.model.device
        
        # Initialize optimized infill calculator
        if self.use_optimized_infill:
            self.infill_calculator = OptimizedInfillCalculator(
                self.model, self.tokenizer, self.device, batch_size, clip_inf
            )
        
    def _load_model(self, use_half: bool, use_int8: bool, mixed_precision: bool, gradient_checkpointing: bool):
        """Load the language model and tokenizer with various optimization options."""
        print(f"Loading model: {self.model_name}")
        
        kwargs = {}
        if use_int8:
            kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
        elif use_half and not mixed_precision:
            kwargs = dict(torch_dtype=torch.bfloat16)
        elif mixed_precision:
            # Mixed precision: model weights in half, calculations in float32
            kwargs = dict(torch_dtype=torch.bfloat16)
            print("Using mixed precision: BFloat16 weights, Float32 calculations")
        
        # Handle different model architectures
        if 'mamba' in self.model_name.lower():
            try:
                from transformers import MambaForCausalLM
                model = MambaForCausalLM.from_pretrained(
                    self.model_name, return_dict=True, device_map='auto', **kwargs
                )
            except ImportError:
                raise ImportError("Mamba models require additional dependencies")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, return_dict=True, device_map='auto', **kwargs
            )
        
        model.eval()
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing for memory efficiency")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def calculate_basic_scores(self, text: str) -> Dict[str, float]:
        """Calculate basic perplexity and compression-based scores."""
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        
        loss = outputs.loss
        log_likelihood = -loss.item()
        
        # Compression-based score
        compressed_length = len(zlib.compress(bytes(text, 'utf-8')))
        zlib_score = log_likelihood / compressed_length
        
        return {
            'loss': log_likelihood,
            'zlib': zlib_score
        }
    
    def calculate_mink_scores(self, text: str) -> Dict[str, float]:
        """Calculate Min-k and Min-k++ scores with optional -inf protection."""
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        
        logits = outputs.logits
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        
        # Token log probabilities
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        
        # Apply -inf protection if enabled
        if self.clip_inf:
            token_log_probs = torch.clamp(token_log_probs, min=NUMERICAL_SAFETY['inf_replacement'])
        
        # Min-k scores
        mink_scores = {}
        for ratio in RATIO_THRESHOLDS:
            k_length = int(len(token_log_probs) * ratio)
            # Convert to float32 for numpy operations (handles BFloat16)
            topk = np.sort(token_log_probs.float().cpu().numpy())[:k_length]
            mink_scores[f'mink_{ratio}'] = np.mean(topk).item()
        
        # Min-k++ scores (normalized)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        
        if self.clip_inf:
            mu = torch.clamp(mu, min=NUMERICAL_SAFETY['inf_replacement'])
            sigma = torch.clamp(sigma, min=NUMERICAL_SAFETY['min_sigma'])
        
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        
        if self.clip_inf:
            mink_plus = torch.clamp(mink_plus, 
                                  NUMERICAL_SAFETY['score_clip_min'], 
                                  NUMERICAL_SAFETY['score_clip_max'])
        
        for ratio in RATIO_THRESHOLDS:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.float().cpu().numpy())[:k_length]
            mink_scores[f'mink++_{ratio}'] = np.mean(topk).item()
        
        return mink_scores
    
    @torch.no_grad()
    def calculate_infill_scores(self, text: str) -> Dict[int, List[float]]:
        """
        Calculate infill-based membership inference scores.
        Uses optimized batched processing for significant speedup.
        """
        if self.use_optimized_infill:
            return self.infill_calculator.calculate_infill_scores_batched(text)
        else:
            # Fallback to original implementation for comparison
            return self._calculate_infill_scores_original(text)
    
    @torch.no_grad()
    def _calculate_infill_scores_original(self, text: str) -> Dict[int, List[float]]:
        """
        Original infill calculation method (slower, for comparison).
        """
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
        
        if input_ids.shape[-1] < 3:
            return {length: [] for length in NEXT_TOKEN_LENGTHS}
        
        # Get model predictions
        logits = self.model(input_ids=input_ids, labels=input_ids).logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Calculate mean and variance for normalization
        mu = (probs[0] * log_probs[0]).sum(-1)
        sigma = ((probs[0]) * torch.square(log_probs[0])).sum(-1) - torch.square(mu)
        sigma = torch.nan_to_num(sigma)
        
        all_ratios = {length: [] for length in NEXT_TOKEN_LENGTHS}
        token_count = input_ids.shape[-1]
        
        for i in range(token_count - 1):
            # Find most likely token at position i+1
            top_k_probs = torch.sort(probs[:, i, :], dim=-1, descending=True)
            actual_token = input_ids[:, i + 1]
            most_likely_token = top_k_probs.indices[:, 0]
            
            # Create alternative sequence with most likely token
            alternative_ids = copy.deepcopy(input_ids)
            alternative_ids[:, i + 1] = most_likely_token
            
            # Get predictions for alternative sequence
            alt_logits = self.model(input_ids=alternative_ids, labels=alternative_ids).logits
            alt_probs = F.softmax(alt_logits, dim=-1)
            alt_log_probs = F.log_softmax(alt_logits, dim=-1)
            
            alt_mu = (alt_probs[0] * alt_log_probs[0]).sum(-1)
            alt_sigma = ((alt_probs[0]) * torch.square(alt_log_probs[0])).sum(-1) - torch.square(alt_mu)
            alt_sigma = torch.nan_to_num(alt_sigma)
            
            # Calculate standardized probability ratios
            ratios = {}
            for length in NEXT_TOKEN_LENGTHS:
                # Add safety checks for division by zero
                sigma_safe = torch.clamp(sigma[i].sqrt(), min=NUMERICAL_SAFETY['min_sigma'])
                alt_sigma_safe = torch.clamp(alt_sigma[i].sqrt(), min=NUMERICAL_SAFETY['min_sigma'])
                
                original_score = (log_probs[:, i, actual_token] - mu[i]) / sigma_safe
                alternative_score = (alt_log_probs[:, i, most_likely_token] - alt_mu[i]) / alt_sigma_safe
                ratios[length] = original_score - alternative_score
            
            # Extend ratios for future tokens
            for length in NEXT_TOKEN_LENGTHS:
                for j in range(i, min(i + length, token_count - 3)):
                    future_token = input_ids[:, j + 2]
                    # Use proper normalization for each position with safety checks
                    sigma_j_safe = torch.clamp(sigma[j + 1].sqrt(), min=NUMERICAL_SAFETY['min_sigma'])
                    alt_sigma_j_safe = torch.clamp(alt_sigma[j + 1].sqrt(), min=NUMERICAL_SAFETY['min_sigma'])
                    
                    orig_future = (log_probs[:, j + 1, future_token] - mu[j + 1]) / sigma_j_safe
                    alt_future = (alt_log_probs[:, j + 1, future_token] - alt_mu[j + 1]) / alt_sigma_j_safe
                    
                    # Check for finite values before adding
                    if torch.isfinite(orig_future) and torch.isfinite(alt_future):
                        ratios[length] += orig_future - alt_future
                
                all_ratios[length].append(ratios[length].item())
        
        return all_ratios
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Run all analysis methods on a single text."""
        scores = {}
        
        # Basic scores
        try:
            basic_scores = self.calculate_basic_scores(text)
            scores.update(basic_scores)
        except Exception as e:
            print(f"Warning: Basic scores failed: {e}")
            scores.update({'loss': 0.0, 'zlib': 0.0})
        
        # Min-k scores
        try:
            mink_scores = self.calculate_mink_scores(text)
            scores.update(mink_scores)
        except Exception as e:
            print(f"Warning: Min-k scores failed: {e}")
            # Add default values for all mink methods
            for ratio in RATIO_THRESHOLDS:
                scores[f'mink_{ratio}'] = 0.0
                scores[f'mink++_{ratio}'] = 0.0
        
        # Optimized infill scores
        try:
            infill_scores = self.calculate_infill_scores(text)
            for infill_length in infill_scores:
                all_probs = np.nan_to_num(infill_scores[infill_length])
                
                # Skip if no scores were calculated
                if len(all_probs) == 0:
                    continue
                    
                for ratio in RATIO_THRESHOLDS:
                    k_length = int(len(all_probs) * ratio)
                    if k_length > 0:
                        topk = np.sort(all_probs)[:k_length]
                        score_value = np.mean(topk).item()
                        # Check for invalid values
                        if np.isfinite(score_value):
                            scores[f'infill_{infill_length}_{ratio}'] = score_value
                        else:
                            scores[f'infill_{infill_length}_{ratio}'] = 0.0
                    else:
                        scores[f'infill_{infill_length}_{ratio}'] = 0.0
        except Exception as e:
            print(f"Warning: Infill scores failed: {e}")
            # Add default values for all infill methods
            for length in NEXT_TOKEN_LENGTHS:
                for ratio in RATIO_THRESHOLDS:
                    scores[f'infill_{length}_{ratio}'] = 0.0
        
        # Final check: ensure all scores are finite
        for key, value in scores.items():
            if not np.isfinite(value):
                print(f"Warning: Invalid score for {key}: {value}, setting to 0.0")
                scores[key] = 0.0
        
        return scores

