import os
import numpy as np
import random
import math
import json
from functools import partial
from PIL import Image
import pandas as pd
from transformers import (
    AutoImageProcessor, 
    DinatForImageClassification, 
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    top_k_accuracy_score, 
    confusion_matrix
)
# CSV Logger utility
import sys
import csv
from datetime import datetime
from torch.utils.data import DataLoader, Sampler, Dataset
from typing import Dict
import logging
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset, concatenate_datasets

from transformers import Trainer
from torchvision.transforms import (
    Compose, 
    Resize, 
    ToTensor
)

# Suppress warnings and configure logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger("natten.functional").setLevel(logging.ERROR)

# Set device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
column = 'label'

# Filter function for data preprocessing
def filter_m_examples(example):
    return example["label"] != 4 and example["label"] != 5 

new_size = 224
size = 224
windows = [
    (0, 0, 112, 147),       # Top-MID left

    (0, 0, 112, 75),       # Top-left
    (112, 0, 224, 75),     # Top-right

    (0, 75, 112, 147),     # mid left
    (112, 75, 224, 147),     # Middle-right


    (0, 149, 112, 224),   # bot left
    (112, 149, 224, 224),   # bot right
    None , 
    None, 
    None                   # Entire image
]

class RandomWindowCrop:
    def __init__(self, windows, output_size):
        self.windows = windows
        self.output_size = output_size

    def __call__(self, img):
        window = random.choice(self.windows)
        if window is not None:
            cropped_img = img.crop(window)
        else:
            cropped_img = img
        return cropped_img.resize((self.output_size, self.output_size), Image.BILINEAR)


_train_transforms = Compose([
    Resize((new_size, new_size)),
    # RandomWindowCrop(windows, size),
    ToTensor(),
])

_val_transforms = Compose([
    Resize((new_size, new_size)),
    ToTensor(),
])

_test_transforms = Compose([
    Resize((new_size, new_size)),
    ToTensor(),
])

# Transform application functions
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def test_transforms(examples):
    examples['pixel_values'] = [_test_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Dataset and sampler classes for training
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def _create_group_indices(self, shuffled_indices):
        group_indices = {}
        for idx in shuffled_indices:
            speaker_id = self.data_source[idx]['speakerID']
            if speaker_id not in group_indices:
                group_indices[speaker_id] = []
            group_indices[speaker_id].append(idx)
        return list(group_indices.values())

    def __iter__(self):
        # Shuffle the entire dataset initially
        shuffled_indices = list(range(self.num_samples))
        random.shuffle(shuffled_indices)
        
        # Group the shuffled indices by speakerID
        self.group_indices = self._create_group_indices(shuffled_indices)
        
        # Shuffle the groups
        random.shuffle(self.group_indices)
        
        # Flatten indices after shuffling groups
        final_indices = [idx for group in self.group_indices for idx in group]
        return iter(final_indices)

    def __len__(self):
        return self.num_samples

# Data processing functions
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[column] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Convert probabilities to class predictions using argmax
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predicted_classes)
    uar = recall_score(labels, predicted_classes, average='macro')
    f1 = f1_score(labels, predicted_classes, average='macro')
    
    return {
        'accuracy': accuracy, 
        'uar': uar, 
        'f1': f1,
    }

# Dataset balancing function
def balance_dataset(dataset, label_column="label", seed=None):
    """
    Balances a Hugging Face dataset by undersampling or oversampling each class to the average number of examples.
    """
    if seed is not None:
        random.seed(seed)
    
    # Group indices by class label
    class_to_indices = {}
    for idx, example in enumerate(dataset):
        class_label = example[label_column]
        if class_label not in class_to_indices:
            class_to_indices[class_label] = []
        class_to_indices[class_label].append(idx)
    
    # Compute the average count across all classes
    total_examples = sum(len(indices) for indices in class_to_indices.values())
    num_classes = len(class_to_indices)
    avg_count = total_examples // num_classes
    
    # Resample indices for each class to reach the average count
    balanced_indices = []
    for label, indices in class_to_indices.items():
        current_count = len(indices)
        if current_count < avg_count:
            # Oversample with replacement
            extra_indices = random.choices(indices, k=(avg_count - current_count))
            balanced_indices.extend(indices + extra_indices)
        elif current_count > avg_count:
            # Undersample without replacement
            selected_indices = random.sample(indices, avg_count)
            balanced_indices.extend(selected_indices)
        else:
            balanced_indices.extend(indices)
    
    # Shuffle indices to mix the classes
    random.shuffle(balanced_indices)
    
    # Create and return a new balanced dataset
    balanced_dataset = dataset.select(balanced_indices)
    return balanced_dataset

# Model implementations
class CustomDinatForImageClassification_V2(nn.Module):
    def __init__(self, base_model, num_classes, feature_dim, class_weights=None, alpha=1.0, beta=0.1, center_lr=0.5):
        """
        base_model: A pretrained image model with .logits as output
        num_classes: Number of classes for classification
        feature_dim: Dimensionality of feature vectors from the base model
        class_weights: Optional weights for the classes
        alpha: Weight for Cross-Entropy Loss
        beta: Weight for Contrastive-Center Loss
        center_lr: Learning rate for updating centers
        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_weights = class_weights

        # Loss function
        self.loss_fn = CombinedLossV2(
            num_classes=num_classes,
            feature_dim=feature_dim,
            alpha=alpha,
            beta=beta,
            center_lr=center_lr, 
            class_weights=class_weights
        )

    def forward(self, pixel_values, labels=None, **kwargs):
        """
        pixel_values: images after transforms (B, C, H, W)
        labels: integer labels (B,)
        """
        # Extract features and logits from the base model
        outputs = self.base_model(pixel_values, **kwargs)
        logits = outputs.logits

        # Get features from the penultimate layer
        features = outputs.hidden_states[-1]
        features = features.mean(dim=(1, 2))

        if labels is not None:
            # Compute combined loss
            loss = self.loss_fn(logits, features, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, alpha=0.5, beta=0.5, center_lr=0.5):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.center_lr = center_lr

        # Initialize class centers
        self.register_buffer('centers', torch.randn(num_classes, feature_dim))

    def forward(self, logits, features, targets):
        # Handle sequence-based features (e.g., transformers)
        if len(features.shape) == 3:  # (B, seq_len, feature_dim)
            features = features.mean(dim=1)  # Pool over sequence length

        # Compute Cross-Entropy Loss
        ce_loss = F.cross_entropy(logits, targets)

        # Gather class centers
        centers_batch = self.centers[targets]

        # Compute Contrastive-Center Loss
        intra_distances = torch.norm(features - centers_batch, p=2, dim=1).mean()
        inter_distances = torch.norm(self.centers.unsqueeze(0) - self.centers.unsqueeze(1), p=2, dim=2)
        inter_distances = inter_distances + torch.eye(self.num_classes, device=inter_distances.device) * 1e12
        inter_distances = inter_distances.min(dim=1)[0].mean()

        contrastive_center_loss = intra_distances - inter_distances

        # Update class centers
        with torch.no_grad():
            diff = centers_batch - features.detach()
            unique_labels, counts = targets.unique(return_counts=True)
            for label, count in zip(unique_labels, counts):
                mask = (targets == label)
                diff_sum = diff[mask].sum(dim=0)
                self.centers[label] -= self.center_lr * diff_sum / count

        # Combine Losses
        return self.alpha * ce_loss + self.beta * contrastive_center_loss

# Custom trainer implementation
class CustomTrainer(Trainer):
    def __init__(self, *args, custom_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_sampler = custom_sampler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        if self.custom_sampler is not None:
            sampler = self.custom_sampler
        else:
            # fallback to default PyTorch random sampler if not provided
            sampler = torch.utils.data.RandomSampler(self.train_dataset)
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        This calls model(**inputs). The model returns {"loss": loss, "logits": logits}.
        """
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss



# Utility functions for saving results
def save_model_header(new_model_path, model_info):
    os.makedirs(new_model_path, exist_ok=True)

    # Define the file path
    file_path = os.path.join(new_model_path, 'header.txt')

    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Write the content to the file
    with open(file_path, 'w') as file:
        file.write(f"Date: {current_date}\n")
        for key, value in model_info.items():
            file.write(f"{key}: {value}\n")

    print(f"File saved successfully at: {file_path}")
    return file_path

def save_confusion_matrix(outputs, dataset_train, new_model_path, Map2Num):
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the confusion matrix with percentages as color scaling
    im = ax.imshow(cm_percentage, interpolation='nearest', cmap='PuBuGn')
    
    # Add color bar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', rotation=270, labelpad=15)
    
    # Set the labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=Map2Num, yticklabels=Map2Num,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            percent = cm_percentage[i, j]
            text = f"{value}\n({percent:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color="black" if percent < 50 else "white")
    
    # Calculate accuracy and UAR
    accuracy = outputs.metrics['test_accuracy'] * 100
    uar = outputs.metrics['test_uar'] * 100
    
    # Save the figure
    filename = f"{os.path.split(dataset_train)[1]}_accuracy_{accuracy:.2f}_UAR_{uar:.2f}.png"
    save_path = os.path.join(new_model_path, 'results')
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    
    print(f"Confusion matrix saved to: {full_path}")
    return full_path

def create_unique_output_dir(base_output_dir: str) -> str:
    """
    Creates a unique output directory appended with the current date and an incremented identifier.
    
    Args:
        base_output_dir (str): The base directory where the new folder should be created.
        
    Returns:
        str: The path of the newly created unique output directory.
    """
    # Get the current date in YYYYMMDD format
    date_str = datetime.now().strftime("%Y%m%d")

    # Get a list of existing directories in the base output directory
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    existing_dirs = [
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d))
    ]

    # Filter for directories that start with the current date string
    matching_dirs = [
        d for d in existing_dirs
        if d.startswith(date_str) and "_" in d and d.split("_")[-1].isdigit()
    ]

    # Determine the next numerical identifier
    if matching_dirs:
        last_num = max(int(d.split("_")[-1]) for d in matching_dirs)
        new_num = last_num + 1
    else:
        new_num = 1

    # Construct the new unique directory path
    unique_output_dir = os.path.join(base_output_dir, f"{date_str}_{new_num}")

    # Create the directory
    os.makedirs(unique_output_dir, exist_ok=True)

    return unique_output_dir


class CombinedLossV2(nn.Module):
    def __init__(self, num_classes, feature_dim, alpha=0.5, beta=0.5, center_lr=0.5, class_weights=None):
        super(CombinedLossV2, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.center_lr = center_lr
        # Initialize class centers
        self.register_buffer('centers', torch.randn(num_classes, feature_dim))
        
        # Initialize class weights (default: equal weights)
        if class_weights is None:
            self.register_buffer('class_weights', torch.ones(num_classes))
        else:
            self.register_buffer('class_weights', class_weights)
            
    def forward(self, logits, features, targets):
        # Handle sequence-based features (e.g., transformers)
        if len(features.shape) == 3:  # (B, seq_len, feature_dim)
            features = features.mean(dim=1)  # Pool over sequence length
            
        # Compute Cross-Entropy Loss with class weights
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        # Gather class centers
        centers_batch = self.centers[targets]
        
        # Compute Contrastive-Center Loss with class weights
        # Get weights for current batch based on their class
        batch_weights = self.class_weights[targets]
        
        # Apply weights to intra-distances (within-class distances)
        intra_distances = torch.norm(features - centers_batch, p=2, dim=1) * batch_weights
        intra_distances = intra_distances.mean()
        
        # Inter-distances (between-class distances) can remain the same
        # or we could also apply class weights to the inter-distances calculation
        inter_distances = torch.norm(self.centers.unsqueeze(0) - self.centers.unsqueeze(1), p=2, dim=2)
        inter_distances = inter_distances + torch.eye(self.num_classes, device=inter_distances.device) * 1e12
        inter_distances = inter_distances.min(dim=1)[0]
        
        # Optionally weight the inter-distances by class weights
        inter_distances = (inter_distances * self.class_weights).mean()
        
        contrastive_center_loss = intra_distances - inter_distances
        
        # Update class centers with weighted updates
        with torch.no_grad():
            diff = centers_batch - features.detach()  # Detach features for updates
            unique_labels, counts = targets.unique(return_counts=True)
            for label, count in zip(unique_labels, counts):
                # Create a mask to identify samples of the current label
                mask = (targets == label)  # Shape: (batch_size,)
                diff_sum = diff[mask].sum(dim=0)  # Correctly index diff with mask
                
                # Apply class weight to the update
                weight_factor = self.class_weights[label]
                self.centers[label] -= self.center_lr * diff_sum * weight_factor / count
                
        # Combine Losses
        return self.alpha * ce_loss + self.beta * contrastive_center_loss
        