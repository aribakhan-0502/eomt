# data/mvtec_dataset.py - CORRECT VERSION
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from torchvision import tv_tensors

class MVTecAnomalyDataset(Dataset):
    def __init__(self, root_dir, category='bottle', split='train', image_size=640):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.image_size = image_size
        
        # Get all image paths
        self.image_paths = []
        self.labels = []  # 0=normal, 1=anomalous
        self.mask_paths = []  # For anomaly localization
        
        category_path = os.path.join(root_dir, category, split)
        
        if split == 'train':
            # Only normal images in training
            good_path = os.path.join(category_path, 'good')
            for img_name in os.listdir(good_path):
                if img_name.endswith('.png'):
                    self.image_paths.append(os.path.join(good_path, img_name))
                    self.labels.append(0)  # Normal
                    self.mask_paths.append(None)
                    
        else:  # test split
            # Good (normal) images
            good_path = os.path.join(category_path, 'good')
            for img_name in os.listdir(good_path):
                if img_name.endswith('.png'):
                    self.image_paths.append(os.path.join(good_path, img_name))
                    self.labels.append(0)
                    self.mask_paths.append(None)
            
            # Defective (anomalous) images
            for defect_type in os.listdir(category_path):
                if defect_type != 'good':
                    defect_path = os.path.join(category_path, defect_type)
                    if os.path.isdir(defect_path):
                        for img_name in os.listdir(defect_path):
                            if img_name.endswith('.png') and not img_name.endswith('_mask.png'):
                                self.image_paths.append(os.path.join(defect_path, img_name))
                                self.labels.append(1)
                                
                                # Find corresponding mask
                                mask_name = img_name.replace('.png', '_mask.png')
                                mask_path = os.path.join(root_dir, category, 'ground_truth', defect_type, mask_name)
                                self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)
        
        print(f"Loaded {len(self.image_paths)} images for {category} {split} split")
        print(f"Normal: {self.labels.count(0)}, Anomalous: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Convert to tensor and resize (matching EoMT preprocessing)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0), 
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Load mask if available
        masks = []
        labels = []
        is_crowd = []
        
        if self.mask_paths[idx] is not None and os.path.exists(self.mask_paths[idx]):
            mask = Image.open(self.mask_paths[idx]).convert('L')
            mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='nearest'
            ).squeeze(0).squeeze(0)
            mask_tensor = (mask_tensor > 0.5).long()
            
            masks.append(mask_tensor)
            labels.append(self.labels[idx])  # Use 1 for anomaly
            is_crowd.append(0)
        else:
            # For normal images or images without masks, create empty mask
            empty_mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            masks.append(empty_mask)
            labels.append(self.labels[idx])
            is_crowd.append(0)
        
        # Stack masks and create target dict (matching EoMT format)
        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
        }
        
        return image_tensor, target