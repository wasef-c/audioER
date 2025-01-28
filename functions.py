import os
import numpy as np
import random
import math
import json
from functools import partial
from PIL import Image
import pandas as pd
from transformers import AutoImageProcessor, ViTForImageClassification, ViTHybridForImageClassification, BeitForImageClassification,DinatForImageClassification, ViTImageProcessor, ConvNextV2ForImageClassification
from sklearn.metrics import accuracy_score, recall_score, f1_score, top_k_accuracy_score, mean_squared_error
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from typing import Dict  # Add this import
import logging
import random
from transformers import TrainerCallback

import torch
from torch import nn
## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## tqdm for loading bars
from tqdm.notebook import tqdm
from datasets import load_dataset, concatenate_datasets, DatasetDict

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.npya as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision import transforms
from transformers import EarlyStoppingCallback

from transformers import Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from scipy.special import lambertw  # Add this import at the top of your file

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
column = 'label'


def filter_m_examples(example):
    return example["label"] != 4 and example["label"] != 5 

import random
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor
)

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

new_size = 224
size = 224

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
from torchvision.transforms import Compose, RandomResizedCrop, RandomRotation, CenterCrop, ColorJitter, ToTensor, Resize, RandomApply

def get_random_crop_size(min_crop=0.5, max_crop=1.0):
    return random.uniform(min_crop, max_crop)

# Create the new train and validation transform pipelines
_train_transforms = Compose(
    # [
    #     Resize((new_size, new_size)),
    #     # RandomWindowCrop(windows, size),

    #     ToTensor(),
    #     # normalize,
    # ]
    [
        # RandomApply([
        #         RandomResizedCrop(
        #             size=224, 
        #             scale=(0.3, 1.0),  # Random scale from 30% to 100% of original size
        #             ratio=(0.75, 1.3333)  # Random aspect ratio
        #         )
        #     ], p=0.4),
        # RandomApply([
        #     ColorJitter(
        #         brightness=(0.5, 3),  # Random brightness from 0 to 0.8
        #         contrast=(0.5, 3),    # Random contrast from 0 to 0.8
        #         saturation=(0.5, 3),  # Random saturation from 0 to 0.8
        #         # hue=(-0.5, 0.5)       # Random hue from -0.5 to 0.5
        #     )
        # ], p=0.8),
        # RandomWindowCrop(windows, size),
        # RandomApply([
        #     lambda img: CenterCrop(size=int(get_random_crop_size() * min(img.size)))(img)
        # ], p=0.5),  # 50% chance of applying a random-sized center crop
        Resize((224,224)),  # Always apply resize to ensure consistent dimensions
        ToTensor()
    ]
)

_val_transforms = Compose(
    [
        Resize((new_size, new_size)),
        ToTensor(),
        # normalize,
    ]
)

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples



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


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[column] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Convert probabilities to class predictions using argmax
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Initialize weights as 1 for all samples
    weights = np.ones_like(labels)
    accuracy= accuracy_score(labels, predicted_classes)
    
    # Compute weighted UAR (Unweighted Average Recall)
    uar = recall_score(labels, predicted_classes, average='macro')
    
    # Compute weighted F1 score
    f1 = f1_score(labels, predicted_classes, average='macro')
    kacc = top_k_accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy, 
        'uar': uar, 
        'f1': f1,
        'top_k_acc': kacc,
    }
class SuperLoss(nn.Module):
    def __init__(self, C=10, lam=1, batch_size=128, class_weights=None):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam
        self.batch_size = batch_size
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
        else:
            sample_weights = torch.ones_like(targets, dtype=torch.float)
        
        l_i = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights).detach()
        sigma = self.sigma(l_i)
        loss = (F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights) - self.tau) * sigma + self.lam * (torch.log(sigma)**2)
        loss = (loss * sample_weights).sum() / self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
        x = x.cuda()  # Ensure it matches the device (GPU)
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()  # Ensure it matches the device (GPU)
        return sigma
    
    import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(train_dataset, class_weight_multipliers):
    labels = [sample['label'] for sample in train_dataset]
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    for class_label, multiplier in class_weight_multipliers.items():
        if class_label in class_weight_dict:
            class_weight_dict[class_label] *= multiplier
    
    return [class_weight_dict[label] for label in unique_classes]


class SuperTrainer(Trainer):
    def __init__(self, *args, super_loss_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.custom_sampler = custom_sampler
        # Initialize SuperLoss with provided parameters or default values
        if super_loss_params is None:
            super_loss_params = {'C': 10, 'lam': 1, 'batch_size': self.args.train_batch_size}
        self.super_loss = SuperLoss(**super_loss_params)

        logging.getLogger().addHandler(logging.NullHandler())
        
        # Disable the natten.functional logger
        logging.getLogger("natten.functional").setLevel(logging.ERROR)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        # Get logits and labels from inputs
        outputs = model(**inputs)
        logits = outputs.get('logits')
        labels = inputs.get('labels')

        # Compute the loss using SuperLoss
        loss = self.super_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Override the log method to filter out unwanted messages
        """
        filtered_logs = {k: v for k, v in logs.items() if "natten.functional" not in str(k)}
        super().log(filtered_logs)

_test_transforms = Compose(
    [
        Resize((new_size, new_size)),
        # RandomWindowCrop(windows, size),
        # Resize((new_size, new_size)),
        ToTensor(),
        # normalize,
    ]
)


def test_transforms(examples):
    examples['pixel_values'] = [_test_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

import os
from datetime import datetime

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
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Map2Num)
    disp.plot(ax=ax, xticks_rotation=45, cmap='PuBuGn')
    
    plt.title('Confusion Matrix')
    
    accuracy = outputs.metrics['test_accuracy'] * 100
    uar = outputs.metrics['test_uar'] * 100
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

# Example usage:
# matrix_path = save_confusion_matrix(outputs, dataset_train, new_model_path)
\


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.special import lambertw

# class SuperLoss(nn.Module):
#     def __init__(self, C=10, lam=1, batch_size=128, class_weights=None):
#         super(SuperLoss, self).__init__()
#         self.tau = math.log(C)
#         self.lam = lam
#         self.batch_size = batch_size
#         self.class_weights = class_weights  # Initial class weights

#     def forward(self, logits, targets, dynamic_weights=None):
#         # Use dynamic_weights if provided, else fallback to fixed class weights
#         if dynamic_weights is not None:
#             sample_weights = dynamic_weights[targets]
#         elif self.class_weights is not None:
#             sample_weights = self.class_weights[targets]
#         else:
#             sample_weights = torch.ones_like(targets, dtype=torch.float)

#         l_i = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights).detach()
#         sigma = self.sigma(l_i)
#         loss = (F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights) - self.tau) * sigma + self.lam * (torch.log(sigma) ** 2)
#         loss = (loss * sample_weights).sum() / self.batch_size
#         return loss

#     def sigma(self, l_i):
#         x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
#         x = x.cuda()  # Ensure it matches the device (GPU)
#         y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
#         y = y.cpu().numpy()
#         sigma = np.exp(-lambertw(y))
#         sigma = sigma.real.astype(np.float32)
#         sigma = torch.from_numpy(sigma).cuda()  # Ensure it matches the device (GPU)
#         return sigma


# class CustomDinatForImageClassification(nn.Module):
#     def __init__(self, base_model, num_classes, initial_class_weights):
#         super().__init__()
#         self.base_model = base_model
#         self.num_classes = num_classes
#         # Initialize learnable class weights
#         self.class_weight_multipliers = nn.Parameter(initial_class_weights.clone())
#         self.class_weights_optimizer = torch.optim.Adam([self.class_weight_multipliers], lr=1e-6)

#         self.loss_fn = SuperLoss(C=num_classes, class_weights=None)  # Initialize without fixed weights

#     def forward(self, pixel_values , labels=None,**kwargs):
#         outputs = self.base_model(pixel_values, **kwargs)
#         logits = outputs.logits
        
#         if labels is not None:
            
#             # Normalize learnable class weights
#             dynamic_weights = torch.softmax(self.class_weight_multipliers, dim=0)
            

#             # Calculate loss using SuperLoss
#             loss = self.loss_fn(logits, labels, dynamic_weights=dynamic_weights)
#             self.class_weights_optimizer.zero_grad()
#             # loss.backward()  # Compute gradients
#             self.class_weights_optimizer.step()  # Update class weights

#             return {"loss": loss, "logits": logits}

#         return {"logits": logits, "class_weights":self.class_weight_multipliers }

# class CustomTrainer(Trainer):
#     def __init__(self, *args, custom_sampler=None, super_loss_params=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.custom_sampler = custom_sampler
#         # Initialize SuperLoss with provided parameters or default values
#         if super_loss_params is None:
#             super_loss_params = {'C': 10, 'lam': 1, 'batch_size': self.args.train_batch_size}
#         self.super_loss = SuperLoss(**super_loss_params)
        

#         logging.getLogger().addHandler(logging.NullHandler())
        
#         # Disable the natten.functional logger
#         logging.getLogger("natten.functional").setLevel(logging.ERROR)
        
#     def get_train_dataloader(self):
#         if self.train_dataset is None:
#             raise ValueError("Trainer: training requires a train_dataset.")

#         # Initialize the custom sampler for each epoch
#         custom_sampler = CustomSampler(self.train_dataset)
#         return DataLoader(
#             self.train_dataset,
#             sampler=custom_sampler,
#             batch_size=self.args.train_batch_size,
#             collate_fn=self.data_collator,
#             drop_last=self.args.dataloader_drop_last,
#             num_workers=self.args.dataloader_num_workers,
#         )

#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.
#         """
#         # Forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         labels = inputs.get("labels")

#         # Retrieve dynamic class weights from the model
#         if hasattr(model, "class_weight_multipliers"):
#             dynamic_weights = torch.softmax(model.class_weight_multipliers, dim=0)
#         else:
#             dynamic_weights = None

#         # Compute the loss using SuperLoss
#         loss = self.super_loss(logits, labels, dynamic_weights=dynamic_weights)

#         return (loss, outputs) if return_outputs else loss

    
#     def log(self, logs: Dict[str, float]) -> None:
#         """
#         Override the log method to filter out unwanted messages
#         """
#         filtered_logs = {k: v for k, v in logs.items() if "natten.functional" not in str(k)}
#         super().log(filtered_logs)
#     def on_epoch_end(self):
#         super().on_epoch_end()
        
#         # Access the class weights from the model
#         if hasattr(self.model, "class_weight_multipliers"):
#             class_weights = self.model.class_weight_multipliers
#             # Convert to softmax to get normalized weights
#             normalized_weights = torch.softmax(class_weights, dim=0)
#             print(f"Class weights at the end of epoch: {normalized_weights.cpu().detach().numpy()}")
#         else:
#             print("not working")


# class ClassWeightLoggerCallback(TrainerCallback):
#     def on_epoch_end(self, args, state, control, model=None, **kwargs):
#         if hasattr(model, "class_weight_multipliers"):
#             class_weights = model.class_weight_multipliers
#             # Convert to softmax to get normalized weights
#             normalized_weights = torch.softmax(class_weights, dim=0)
#             print(f"Class weights at the end of epoch: {normalized_weights.cpu().detach().numpy()}")
#         else:
#             print("Class weights not found in the model.")


# import sys
# import csv
# from datetime import datetime

# class CSVLogger:
#     def __init__(self, log_file, fieldnames):
#         self.console = sys.stdout
#         self.log_file = log_file
#         self.fieldnames = fieldnames
        
#         # Initialize CSV file and write the header
#         with open(self.log_file, mode='w', newline='') as file:
#             writer = csv.DictWriter(file, fieldnames=self.fieldnames)
#             writer.writeheader()

#     def write(self, message):
#         # Write to console
#         self.console.write(message)

#         # Write to CSV (only non-empty lines)
#         if message.strip():
#             with open(self.log_file, mode='a', newline='') as file:
#                 writer = csv.DictWriter(file, fieldnames=self.fieldnames)
#                 writer.writerow({"timestamp": datetime.now(), "log": message.strip()})

#     def flush(self):
#         # Flush is needed for proper console behavior
#            pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     """
#     A simple focal loss implementation that optionally makes alpha/gamma learnable.
#     No class weighting is used here.
#     """
#     def __init__(self, alpha_init=1.0, gamma_init=2.0, learnable=True):
#         super().__init__()
        
#         if learnable:
#             # alpha and gamma as learnable parameters
#             self.alpha = nn.Parameter(torch.tensor(alpha_init, requires_grad=True))
#             self.gamma = nn.Parameter(torch.tensor(gamma_init, requires_grad=True))
#         else:
#             # fixed scalars (registered as buffers for consistency)
#             self.register_buffer("alpha", torch.tensor(alpha_init))
#             self.register_buffer("gamma", torch.tensor(gamma_init))

#     def forward(self, logits, targets):
#         """
#         :param logits:  (B, num_classes)
#         :param targets: (B,) - integer labels
#         :return: scalar focal loss
#         """
#         # Standard CE (per-sample, no reduction)
#         ce_loss = F.cross_entropy(logits, targets, reduction='none')  # shape: (B,)

#         # pt = exp(-ce_loss) in focal-loss literature
#         pt = torch.exp(-ce_loss)
        
#         # focal multiplier
#         focal_factor = (1.0 - pt) ** self.gamma
        
#         # combine
#         focal_loss = self.alpha * focal_factor * ce_loss
        
#         return focal_loss.mean()

class AdaptiveLearnableFocalLoss_V2(nn.Module):
    """
    A focal loss variant with:
      - Class-dependent alpha and gamma (vector form)
      - Exponential reparameterization to keep alpha, gamma > 0
      - Optional mixture of focal loss and cross-entropy via adaptive_factor
    """
    def __init__(self,
                 num_classes: int,
                 alpha_init: float = 1.0,
                 gamma_init: float = 2.0,
                 learnable: bool = True,
                 class_weights: torch.Tensor = None):
        super(AdaptiveLearnableFocalLoss_V2, self).__init__()

        self.num_classes = num_classes
        self.class_weights = class_weights

        # We store log(alpha) and log(gamma) as parameters for stability
        if learnable:
            # Initialize all classes to alpha_init, gamma_init
            self.log_alpha = nn.Parameter(
                torch.log(torch.ones(num_classes) * alpha_init)
            )
            self.log_gamma = nn.Parameter(
                torch.log(torch.ones(num_classes) * gamma_init)
            )
        else:
            # If not learnable, store them as buffers (non-trainable)
            # and just keep a constant vector for alpha, gamma
            alpha_vec = torch.ones(num_classes) * alpha_init
            gamma_vec = torch.ones(num_classes) * gamma_init

            self.register_buffer('alpha', alpha_vec)
            self.register_buffer('gamma', gamma_vec)
            self.log_alpha = None
            self.log_gamma = None

        # Adaptive mix factor in [0,1] between focal loss and cross-entropy
        self.adaptive_factor = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size) with class indices
        Returns:
            Scalar loss (averaged over batch).
        """

        # 1. Compute cross-entropy loss (element-wise)
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(
                logits,
                targets,
                reduction='none',
                weight=self.class_weights.to(logits.device)
            )
        else:
            ce_loss = F.cross_entropy(
                logits,
                targets,
                reduction='none'
            )

        # 2. Convert CE loss to p_t = Probability assigned to the true class
        #    In standard CE:  ce_loss = -log(p_t)  =>  p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # 3. Retrieve alpha, gamma (class-dependent) by indexing with targets
        if self.log_alpha is not None and self.log_gamma is not None:
        # Exponential reparam => alpha, gamma > 0
            alpha_vec = torch.exp(self.log_alpha).to(logits.device)
            gamma_vec = torch.exp(self.log_gamma).to(logits.device)
        
            alpha_t = alpha_vec[targets]      # Now both alpha_vec and targets are on the same device
            gamma_t = gamma_vec[targets]
        else:
            # Non-learnable path
            alpha_t = self.alpha.to(logits.device)[targets]
            gamma_t = self.gamma.to(logits.device)[targets]

        # 4. Compute focal term: (1 - p_t)^gamma_t
        focal_term = (1.0 - pt) ** gamma_t
        
        # 5. Focal loss (element-wise)
        focal_loss = alpha_t * focal_term * ce_loss

        # 6. Mix focal loss and cross-entropy via an adaptive factor in [0,1]
        #    We clamp to ensure stable mixing between the two.
        adaptive_factor_clamped = torch.clamp(self.adaptive_factor, 0.0, 1.0)
        combined_loss = adaptive_factor_clamped * focal_loss \
                        + (1.0 - adaptive_factor_clamped) * ce_loss
        
        # 7. Return average over the batch
        return combined_loss.mean()
    
class FocalLoss(nn.Module):
    """
    Focal Loss implementation with optional learnable alpha/gamma and class weights.
    """
    def __init__(self, alpha_init=1.0, gamma_init=2.0, class_weights=None, learnable=True):
        super().__init__()
        
        if learnable:
            # alpha and gamma as learnable parameters
            self.alpha = nn.Parameter(torch.tensor(alpha_init, requires_grad=True))
            self.gamma = nn.Parameter(torch.tensor(gamma_init, requires_grad=True))
        else:
            # fixed scalars (registered as buffers for consistency)
            self.register_buffer("alpha", torch.tensor(alpha_init))
            self.register_buffer("gamma", torch.tensor(gamma_init))
        
        # Store class weights
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights))
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        :param logits:  (B, num_classes)
        :param targets: (B,) - integer labels
        :return: scalar focal loss
        """
        # Standard CE (per-sample, no reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights)  # shape: (B,)

        # pt = exp(-ce_loss) in focal-loss literature
        pt = torch.exp(-ce_loss)
        
        # focal multiplier
        focal_factor = (1.0 - pt) ** self.gamma
        
        # Combine focal factor, alpha, and ce_loss
        focal_loss = self.alpha * focal_factor * ce_loss
        
        return focal_loss.mean()


class CustomDinatForImageClassification(nn.Module):
    def __init__(self, base_model, num_classes, class_weights = None):
        """
        base_model: A pretrained image model with .logits as output
        num_classes: # of classes for classification
        """
        super().__init__()
        self.base_model = base_model  # e.g. a DinAT or ViT model
        self.num_classes = num_classes
        self.class_weights = class_weights
        
        # Create your focal loss instance:
        self.loss_fn = AdaptiveLearnableFocalLoss_V2(
            # alpha_init=1.0,
            # gamma_init=2.0,
            num_classes = self.num_classes,
            class_weights = class_weights,
            learnable=True
        )
        # self.loss_fn = FocalLoss(
        #     alpha_init=1.0,
        #     gamma_init=2.0,
        #     class_weights = class_weights,
        #     learnable=True
        # )

    def forward(self, pixel_values, labels=None, **kwargs):
        """
        pixel_values: images after transforms (B, C, H, W)
        labels: integer labels (B,)
        """
        outputs = self.base_model(pixel_values, **kwargs)
        logits = outputs.logits  # shape: (B, num_classes)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)  # focal loss
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

from transformers import Trainer

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


import sys
import csv
from datetime import datetime

class CSVLogger:
    def __init__(self, log_file, fieldnames=None):
        """
        log_file: file path for CSV
        fieldnames: list of columns, e.g. ["timestamp", "log"]
        """
        if fieldnames is None:
            fieldnames = ["timestamp", "log"]
        
        self.console = sys.stdout
        self.log_file = log_file
        self.fieldnames = fieldnames

        # Initialize CSV file and write the header
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def write(self, message):
        # 1) Write to console
        self.console.write(message)

        # 2) Write non-empty lines to CSV
        if message.strip():
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writerow({
                    "timestamp": datetime.now(),
                    "log": message.strip()
                })

    def flush(self):
        pass
