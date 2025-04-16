import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import accuracy_score, recall_score, f1_score, top_k_accuracy_score
from torch.utils.data import Dataset, DataLoader, Sampler
from datetime import datetime
import logging
from scipy.special import lambertw
from sklearn.utils.class_weight import compute_class_weight
from collections import OrderedDict, defaultdict
from transformers import AutoImageProcessor, DinatForImageClassification
# Enhanced augmentations for training
from torchvision import transforms
from torchvision.transforms import (
    Compose, Resize, ToTensor, RandomHorizontalFlip, 
    RandomAffine, ColorJitter, Normalize
)
from PIL import Image

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformations
def get_transforms(new_size=224):
    return Compose([
        Resize((new_size, new_size)),
        ToTensor()
    ])

# Test transforms for validation and testing
_test_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
])

def test_transforms(examples):
    examples['pixel_values'] = [_test_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Train transforms
# _train_transforms = Compose([
#     Resize((224, 224)),
#     ToTensor()
# ])

# # Enhanced augmentations for training
# from torchvision import transforms

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import ColorJitter
class RandomMasking(object):
    """
    Apply random rectangular masks to an image.
    This is particularly useful for acoustic feature images.
    """
    def __init__(self, max_mask_patches=16, mask_ratio=0.1):
        self.max_mask_patches = max_mask_patches  # Maximum number of patches to mask
        self.mask_ratio = mask_ratio  # Percentage of image that can be masked
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be masked
        Returns:
            PIL Image: Masked image
        """
        # Convert to numpy for processing
        img_np = np.array(img).copy()
        
        # Handle both RGB and grayscale images
        h, w = img_np.shape[:2]
        is_rgb = len(img_np.shape) == 3
        
        # Determine number of patches to mask
        n_masks = int(self.mask_ratio * h * w / 64)  # Assuming average patch size ~64 pixels
        n_masks = min(n_masks, self.max_mask_patches)
        
        # Apply random masks
        for _ in range(n_masks):
            # Generate random mask dimensions and position
            height = np.random.randint(4, 9)
            width = np.random.randint(4, 9)
            top = np.random.randint(0, h - height)
            left = np.random.randint(0, w - width)
            
            # Apply the mask (set to zero)
            if is_rgb:
                img_np[top:top+height, left:left+width, :] = 0
            else:
                img_np[top:top+height, left:left+width] = 0
        
        return Image.fromarray(img_np)
    

_train_transforms = Compose([
    Resize((224, 224)),
    # RandomMasking(max_mask_patches=8, mask_ratio=0.05),  # Add random masking
    # ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_val_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

def train_transforms(examples):
    processed_images = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    examples['pixel_values'] = processed_images
    return examples

def val_transforms(examples):
    processed_images = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    examples['pixel_values'] = processed_images
    return examples

# Custom Dataset and Sampler
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

column = "label"

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[column] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Dataset and sampler classes for training
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# GEM Pooling for feature extraction
class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Permute to (batch_size, channels, height, width)
        if x.dim() == 3:
            patch_dim = int((x.size(1) - 1) ** 0.5)  # Calculate grid size
            x = x[:, 1:, :]  # Remove classification token if present
            # Reshape to [batch_size, height, width, channels]
            x = x.view(x.size(0), patch_dim, patch_dim, x.size(2))

        x = x.permute(0, 3, 1, 2)
        # Apply GeM pooling
        pooled = torch.mean(x.clamp(min=self.eps).pow(
            self.p), dim=(2, 3)).pow(1.0 / self.p)
        return pooled

# Loss Functions
class AdaptiveLearnableFocalLoss(nn.Module):
    def __init__(self, alpha_init=1.0, gamma_init=2.0, learnable=True, class_weights=None):
        super(AdaptiveLearnableFocalLoss, self).__init__()

        # Learnable parameters for alpha and gamma
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, requires_grad=True))
            self.gamma = nn.Parameter(torch.tensor(gamma_init, requires_grad=True))
        else:
            self.alpha = torch.tensor(alpha_init)
            self.gamma = torch.tensor(gamma_init)

        # Class weights (passed as input)
        self.class_weights = class_weights

        # Adaptive weighting factor for focal and class-weighted loss
        self.adaptive_factor = nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self, logits, targets):
        # Compute Cross-Entropy Loss with class weights
        if self.class_weights is None:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights.to(logits.device))
        
        # Compute probability of the true class (pt)
        pt = torch.exp(-ce_loss)

        # Compute Focal Loss with learnable alpha and gamma
        focal_term = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_term * ce_loss

        # Adaptive weighting between focal loss and cross-entropy loss
        combined_loss = self.adaptive_factor * focal_loss + (1 - self.adaptive_factor) * ce_loss

        return combined_loss.mean()

# class BalancedCrossEntropyWithContrastiveLoss(nn.Module):
#     """
#     Enhanced loss function that combines cross-entropy and contrastive-center loss
#     with class balancing mechanisms to optimize for UAR and low std deviation.
#     """
#     def __init__(self,
#                  num_classes: int,
#                  feature_dim: int,
#                  class_weight_multipliers: torch.Tensor = None):
#         super(BalancedCrossEntropyWithContrastiveLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feature_dim = feature_dim
#         self.class_weight_multipliers = class_weight_multipliers
#         self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

#         self.log_var_ce = nn.Parameter(torch.tensor([-0.6]))  # CE ~1.4, so log(1.4²) ≈ 0.7
#         self.log_var_contrastive = nn.Parameter(torch.tensor([1.0]))  # Contrastive ~45, so log(45²) ≈ 7.5 
#         self.log_var_balance = nn.Parameter(torch.tensor([-0.6]))  # Balance ~0.3, so log(0.3²) ≈ -2.4
#         self.gamma = nn.Parameter(torch.tensor([5.1]))
#         self.happy_weight = nn.Parameter(torch.tensor([class_weight_multipliers[1]]))
#         self.neutral_weight = nn.Parameter(torch.tensor([class_weight_multipliers[0]]))

        
#     def compute_class_accuracies(self, logits, targets):
#         """Compute per-class accuracies for the current batch."""
#         predictions = torch.argmax(logits, dim=1)
#         class_accs = []
        
#         for c in range(self.num_classes):
#             class_mask = (targets == c)
#             if torch.sum(class_mask) > 0:
#                 class_acc = torch.sum((predictions == targets) & class_mask).float() / torch.sum(class_mask)
#                 class_accs.append(class_acc)
                
#         return torch.stack(class_accs)
    
#     def forward(self, logits, features, targets):
#         device = self.centers.device
#         logits = logits.to(device)
#         features = features.to(device)
#         targets = targets.to(device)
        
        
        
#         # Use inverse frequency weighting if no weights provided
#         class_counts = torch.bincount(targets, minlength=self.num_classes)
#         class_weights = (1.0 / (class_counts + 1)).to(device)
#         class_weights = class_weights / class_weights.sum() * self.num_classes
#         # weight_multiplier = float(self.happy_weight)# Adjust this value as needed
#         # Create a new tensor that maintains gradient connection
#         happy_weight_value = self.happy_weight[0]  # Access the element while keeping gradient connection
#         neutral_weight_value = self.neutral_weight[0]

#         # Apply to class weights
#         class_weights[1] = class_weights[1] * happy_weight_value
#         class_weights[0] = class_weights[0] * neutral_weight_value

#         gamma = self.gamma# Focusing parameter - adjust as needed
        
#         # Compute probabilities
#         probs = F.softmax(logits, dim=1)
#         pt = probs[torch.arange(logits.size(0)), targets]  # Get probability of target class
        
#         # Standard CE loss
#         ce_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
        
#         # Apply focal weighting
#         focal_weight = (1 - pt) ** gamma
#         focal_loss = (focal_weight * ce_loss).mean()

#         # 2. Enhanced Contrastive-Center Loss
#         centers_batch = self.centers[targets]
#         batch_size = features.size(0)

#         # center_loss = torch.sum((features - centers_batch)**2) / batch_size
        
#         # Compute intra-class distances
#         intra_class_distance = torch.norm(features - centers_batch, p=2, dim=1)
        
#         # Compute inter-class distances (push different classes apart)
#         center_distances = []
#         for c in range(self.num_classes):
#             mask = (targets != c)
#             if torch.sum(mask) > 0:
#                 dist = torch.norm(features[mask] - self.centers[c], p=2, dim=1)
#                 center_distances.append(torch.mean(torch.exp(-dist)))
        
#         inter_class_distance = torch.mean(torch.stack(center_distances)) if center_distances else torch.tensor(0.0).to(device)
        
#         # Combined contrastive loss that maximizes inter-class distances
#         contrastive_loss = intra_class_distance.mean() + 0.001 * torch.norm(self.centers, p=2).mean() - torch.log(inter_class_distance + 1e-8)
#         # 3. Class Balance Loss - minimize std deviation of class accuracies
#         class_accuracies = self.compute_class_accuracies(logits, targets)
#         balance_loss = torch.std(class_accuracies)
        
#         # Homoscedastic uncertainty weighting
#         precision_ce = torch.exp(-self.log_var_ce)
#         precision_contrastive = torch.exp(-self.log_var_contrastive)
#         precision_balance = torch.exp(-self.log_var_balance)
        
#         # Weighted losses with learned coefficients
#         weighted_ce = precision_ce * focal_loss + 0.5 * self.log_var_ce
#         weighted_contrastive = precision_contrastive * contrastive_loss + 0.5 * self.log_var_contrastive
#         weighted_balance = precision_balance * balance_loss + 0.5 * self.log_var_balance
        
#         # Total loss
#         total_loss = weighted_ce + weighted_contrastive + weighted_balance
#         # total_loss = weighted_ce + weighted_balance


#         weight_info = {
#             'log_var_ce': self.log_var_ce.item(),
#             'log_var_contrastive': self.log_var_contrastive.item(),
#             'log_var_balance': self.log_var_balance.item(),
#             'weight_ce': precision_ce.item(),
#             'weight_contrastive': precision_contrastive.item(),
#             'weight_balance': precision_balance.item(),
#             'gamma': self.gamma.item(),
#             'happy_weight': self.happy_weight.item(),
#             'neutral_weight': self.neutral_weight.item()
#         }
    
#         return total_loss, weight_info
class BalancedCrossEntropyWithContrastiveLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 class_weight_multipliers: torch.Tensor = None):
        super(BalancedCrossEntropyWithContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_weight_multipliers = class_weight_multipliers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

        self.log_var_ce = nn.Parameter(torch.tensor([-0.5]))
        self.log_var_contrastive = nn.Parameter(torch.tensor([1.6]))
        self.log_var_balance = nn.Parameter(torch.tensor([-0.6]))
        
        # Base gamma parameter for focal loss
        self.base_gamma = nn.Parameter(torch.tensor([5.1]))
        
        # Class-specific adaptive parameters
        # Initialize with different values for each class
        self.class_gammas = nn.Parameter(torch.tensor([1.3,2.0,1.0,1.0]))
        
        # Initialize class weights as learnable parameters
        self.class_weights = nn.Parameter(torch.ones(num_classes))
        if class_weight_multipliers is not None:
            # Initialize directly with provided multipliers
            self.class_weights = nn.Parameter(torch.tensor(class_weight_multipliers))
        else:
            self.class_weights = nn.Parameter(torch.ones(num_classes))
    
    def compute_class_accuracies(self, logits, targets,  class_weights=None):
        """Compute per-class accuracies for the current batch."""
        predictions = torch.argmax(logits, dim=1)
        class_accs = []
        if class_weights is None:
            class_weights = torch.ones(self.num_classes, device=logits.device)
        
        for c in range(self.num_classes):
            class_mask = (targets == c)
            if torch.sum(class_mask) > 0:
                class_acc = torch.sum((predictions == targets) & class_mask).float() / torch.sum(class_mask)
                class_accs.append(class_acc)
                
        return torch.stack(class_accs)
    
    def forward(self, logits, features, targets):
        device = self.centers.device
        logits = logits.to(device)
        features = features.to(device)
        targets = targets.to(device)
        
        # Get frequency-based class weights
        class_counts = torch.bincount(targets, minlength=self.num_classes)
        freq_weights = (1.0 / (class_counts + 1)).to(device)
        freq_weights = freq_weights / freq_weights.sum() * self.num_classes
        
        # Combine with learnable class weights
        # We need to detach for cross_entropy but keep track of computations for gradient flow
        combined_weights = freq_weights * self.class_weights
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)
        pt = probs[torch.arange(logits.size(0)), targets]  # Probability of target class
        
        # Get adaptive gamma for each sample based on its class
        sample_gammas = self.base_gamma * self.class_gammas[targets]
        
        # Compute cross entropy (standard CE with detached weights)
        ce_loss = F.cross_entropy(logits, targets, weight=combined_weights.detach(), reduction='none')
        
        # Apply adaptive focal weighting
        focal_weights = (1 - pt) ** sample_gammas #.detach()  # Detach to stabilize training
        focal_loss = (focal_weights * ce_loss).mean()
        
        # Add a regularization term to avoid extreme values for class_weights
        weight_reg = 0.01 * torch.sum((self.class_weights - 1.0)**2)
        gamma_reg = 0.01 * torch.sum((self.class_gammas - 1.0)**2)
        
        # Rest of your existing loss computation
        # Contrastive loss calculation
        centers_batch = self.centers[targets]
        intra_class_distance = torch.norm(features - centers_batch, p=2, dim=1)
        
        center_distances = []
        for c in range(self.num_classes):
            mask = (targets != c)
            if torch.sum(mask) > 0:
                dist = torch.norm(features[mask] - self.centers[c], p=2, dim=1)
                center_distances.append(torch.mean(torch.exp(-dist)))
        
        inter_class_distance = torch.mean(torch.stack(center_distances)) if center_distances else torch.tensor(0.0).to(device)
        contrastive_loss = intra_class_distance.mean() + 0.001 * torch.norm(self.centers, p=2).mean() - torch.log(inter_class_distance + 1e-8)
        
        # Balance loss calculation
        class_accuracies = self.compute_class_accuracies(logits, targets)
        accuracy_weight_correlation = -torch.sum(class_accuracies * combined_weights[:len(class_accuracies)]) / len(class_accuracies)
        balance_loss = torch.std(class_accuracies)
        balance_loss = balance_loss + 0.1 * accuracy_weight_correlation

        
        # Homoscedastic uncertainty weighting
        precision_ce = torch.exp(-self.log_var_ce)
        precision_contrastive = torch.exp(-self.log_var_contrastive)
        precision_balance = torch.exp(-self.log_var_balance)
        
        # Weighted losses with learned coefficients
        weighted_ce = precision_ce * (focal_loss + weight_reg + gamma_reg) + 0.5 * self.log_var_ce
        weighted_contrastive = precision_contrastive * contrastive_loss + 0.5 * self.log_var_contrastive
        weighted_balance = precision_balance * balance_loss + 0.5 * self.log_var_balance
        
        # Total loss
        total_loss = weighted_ce + weighted_contrastive + weighted_balance
        
        # Output weight info for monitoring
        weight_info = {
            'log_var_ce': self.log_var_ce.item(),
            'log_var_contrastive': self.log_var_contrastive.item(),
            'log_var_balance': self.log_var_balance.item(),
            'weight_ce': precision_ce.item(),
            'weight_contrastive': precision_contrastive.item(),
            'weight_balance': precision_balance.item(),
            'base_gamma': self.base_gamma.item(),
            'class_weights': [w.item() for w in self.class_weights],
            'class_gammas': [g.item() for g in self.class_gammas]
        }
    
        return total_loss, weight_info
# DiNAT model with additional feature extraction
class DiNATWithFeatures(nn.Module):
    def __init__(self, pretrained_model, num_classes, feature_dim=512):
        super(DiNATWithFeatures, self).__init__()
        self.dinat = pretrained_model
        self.gem_pooling = GeMPooling()
        
        # Feature extraction and classification layers
        self.features = nn.Sequential(
            nn.Linear(512, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(768, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Replace the classifier in DiNAT
        self.dinat.classifier = nn.Linear(512, num_classes)

        

    def forward(self, pixel_values):
        outputs = self.dinat(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # Apply GeM pooling
        pooled_features = self.gem_pooling(hidden_states)
        
        # Get feature representation
        features = self.features(pooled_features)
        
        # Forward pass through classifier
        logits = self.dinat.classifier(pooled_features)
        
        return {"logits": logits, "features": features}



# Utility Functions
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predicted_classes)
    uar = recall_score(labels, predicted_classes, average='macro')
    f1 = f1_score(labels, predicted_classes, average='macro')
    kacc = top_k_accuracy_score(labels, predictions, k=2)
    return {'accuracy': accuracy, 'uar': uar, 'f1': f1, 'top_k_acc': kacc}

def calculate_class_weights(train_dataset, class_weight_multipliers=None):
    labels = [sample['label'] for sample in train_dataset]
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    
    class_weight_dict = dict(zip(unique_classes, class_weights))
    if class_weight_multipliers is not None:
        for class_label, multiplier in class_weight_multipliers.items():
            if class_label in class_weight_dict:
                class_weight_dict[class_label] *= multiplier
    
    return [class_weight_dict[label] for label in unique_classes]

def create_unique_output_dir(base_output_dir: str) -> str:
    """
    Creates a unique output directory appended with the current date and an incremented identifier.
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

def balance_dataset(dataset, label_column="label", seed=None):
    """
    Balances a Hugging Face dataset by undersampling or oversampling each class to the average number of examples.
    """
    if seed is not None:
        random.seed(seed)
    
    # 1. Group indices by class label (using the specified label column)
    class_to_indices = defaultdict(list)
    for idx, example in enumerate(dataset):
        class_label = example[label_column]
        class_to_indices[class_label].append(idx)
    
    # 2. Compute the average count across all classes
    total_examples = sum(len(indices) for indices in class_to_indices.values())
    num_classes = len(class_to_indices)
    avg_count = total_examples // num_classes
    
    # 3. Resample indices for each class to reach the average count
    balanced_indices = []
    for label, indices in class_to_indices.items():
        current_count = len(indices)
        if current_count < avg_count:
            extra_indices = random.choices(indices, k=(avg_count - current_count))
            balanced_indices.extend(indices + extra_indices)
        elif current_count > avg_count:
            selected_indices = random.sample(indices, avg_count)
            balanced_indices.extend(selected_indices)
        else:
            balanced_indices.extend(indices)
    
    # Shuffle indices to mix the classes
    random.shuffle(balanced_indices)
    
    # 4. Create and return a new balanced dataset using .select
    balanced_dataset = dataset.select(balanced_indices)
    return balanced_dataset

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
    print(f"Regular Dataset Length: {len(dataset)} -- Balanced Dataset Length: {len(balanced_dataset)}")
    return balanced_dataset


def plot_and_save_confusion_matrix(all_labels, all_predictions, ordered_labels_str, output_dir, epoch=None, filename="confusion_matrix.png"):
    """
    Creates and saves a confusion matrix visualization.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # Step 1: Calculate the confusion matrix
    cm = confusion_matrix(y_true=all_labels, y_pred=all_predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Percentages per class

    # Step 2: Prepare annotations (raw counts and percentages)
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)"

    # Step 3: Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_percentage,  # Use percentage for color scaling
        annot=annotations,
        fmt="",  # Disable default formatting for annotations
        cmap="Blues",
        xticklabels=ordered_labels_str,
        yticklabels=ordered_labels_str,
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (Raw Count & Percentage)")

    # Step 4: Save the plot
    if epoch is None:
        save_path = os.path.join(output_dir, filename)
    else:
        save_path = os.path.join(output_dir, f"Epoch{epoch}_val_confusion_matrix.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()  # Close the figure to free up memory
    print(f"Confusion matrix saved to: {save_path}")

def save_model_info_to_txt(model_info, dirname, filename="results.txt"):
    """
    Save model information dictionary to a text file.
    """
    # Create full file path
    filepath = os.path.join(dirname, filename)
    
    # Write information to text file
    with open(filepath, 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n\n")
    
    print(f"Model information written to {filepath}")