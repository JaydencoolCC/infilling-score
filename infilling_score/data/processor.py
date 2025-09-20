"""
Data processing utilities for WikiMIA datasets.
"""

import re
from typing import List, Dict, Any
from datasets import load_dataset, load_from_disk


class DataProcessor:
    """Handle WikiMIA data loading and preprocessing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove special characters except punctuation
        cleaned = re.sub(r'[^().,a-zA-Z ]', '', text)
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    @staticmethod
    def convert_huggingface_data_to_list_dic(dataset) -> List[Dict[str, Any]]:
        """Convert HuggingFace dataset to list of dictionaries."""
        all_data = []
        for i in range(len(dataset)):
            ex = dataset[i]
            all_data.append(ex)
        return all_data
    
    @staticmethod
    def load_wikimia_data(dataset_name: str) -> List[Dict[str, Any]]:
        """Load and preprocess WikiMIA dataset."""
        print(f"Loading WikiMIA dataset: {dataset_name}")
        
        # Load dataset from HuggingFace
        if not 'paraphrased' in dataset_name:
            dataset = load_dataset('swj0419/WikiMIA', split=dataset_name)
        else:
            dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=dataset_name)
        
        # Convert to list of dictionaries
        data = DataProcessor.convert_huggingface_data_to_list_dic(dataset)
        
        print(f"Loaded {len(data)} samples from WikiMIA")
        return data
    
    def load_reasoning_data(dataset_name: str) -> List[Dict[str, Any]]:
        """Load and preprocess custom reasoning dataset."""
        print(f"Loading reasoning dataset from: {dataset_name}")
        data_config = {
            "s1K_split": "/mnt/sharedata/hdd/users/zhanghx/ssd2/zhanghx/dataset/s1K_tokenized",
            "s1.1K_split": "/mnt/sharedata/hdd/users/zhanghx/ssd2/zhanghx/dataset/s1K-1.1_tokenized",
            "limo_split": "/mnt/sharedata/hdd/users/zhanghx/ssd2/zhanghx/dataset/LIMO_tokenized",
        }
        dataset = load_from_disk(data_config[dataset_name])
        
        if dataset_name in ["s1K_split", "s1.1K_split"]:
            dataset = dataset['train']
        
        train_test_split = dataset.train_test_split(train_size=0.8, seed=42, shuffle=True)
        train_dataset = train_test_split['train']
        test_dataset =  train_test_split['test']
        print(f"train dataset: {len(train_dataset)} test_dataset: {len(test_dataset)}")
        member = train_dataset
        non_member = test_dataset
        sample_num = len(non_member)
        # choose first sample_num from member
        # member = member.select(range(sample_num))
        
        data_dic_list = []
        for i in range(sample_num):
            member_ex = member[i]
            # non_member_ex = non_member[i]
            data_dic_list.append({
                "input": member_ex['question'],
                "label": 1
            })
        for i in range(sample_num):
            non_member_ex = non_member[i]
            data_dic_list.append({
                "input": non_member_ex['question'],
                "label": 0
            })
        # Convert to list of dictionaries
        # data = DataProcessor.convert_huggingface_data_to_list_dic(dataset)
        
        data = data_dic_list
        print(f"Loaded {len(data)} samples from reasoning dataset")
        return data
    
    @staticmethod
    def print_dataset_statistics(data: List[Dict[str, Any]]) -> None:
        """Print statistics about the loaded dataset."""
        if not data:
            print("No data loaded.")
            return
            
        labels = [sample['label'] for sample in data]
        num_train = sum(labels)
        num_non_train = len(labels) - num_train
        
        print(f"Dataset statistics:")
        print(f"  Total samples: {len(data)}")
        print(f"  Training samples: {num_train}")
        print(f"  Non-training samples: {num_non_train}")
        print(f"  Balance ratio: {num_train/len(data):.2%} training")
        
        # Sample length statistics
        if data and 'input' in data[0]:
            lengths = [len(sample['input'].split()) for sample in data[:100]]  # Sample first 100
            avg_length = sum(lengths) / len(lengths)
            print(f"  Average text length (first 100 samples): {avg_length:.1f} words")

