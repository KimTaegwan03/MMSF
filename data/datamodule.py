import os
from typing import Dict, Optional, Tuple
import torch
import torch_geometric
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        device: str = 'cpu'
    ):
        self.root = root
        self.split = split
        self.device = device
        
        self.features = self._load_features() # List[Tensor]
        
    def _load_features(self) -> Dict[str, torch.Tensor]:
        """Load features for each modality"""
        features = {}
        
        # Load text features
        text_path = os.path.join(self.root, f"emb.pt")
        if os.path.exists(text_path):
            features["text"] = torch.load(text_path)
            
        # Load image features
        image_path = os.path.join(self.root, "features_clip.pt")
        if os.path.exists(image_path):
            features["image"] = torch.load(image_path).to(self.device)
            
        return features
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features["text"][index].to(self.device)
    
    def get(self, index):
        return self.features["text"][index].to(self.device)
    
    