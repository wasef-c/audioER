from functionsV2 import *
import logging
import warnings
import pandas as pd
import numpy as np
import os
import torch
import json
from datetime import datetime
from transformers import TrainingArguments, Trainer, SchedulerType, AutoImageProcessor
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import itertools
import csv

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger("natten.functional").setLevel(logging.ERROR)

# Device setup
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Deterministic setup for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA is available. Version: {torch.version.cuda}")
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")

# Paths and Configuration
DATASET_PATH = "../data"
CHECKPOINT_PATH = "../VIT/EMPR"
wt_dc = 0.05
model_type = 'Dinat_Superloss'

# Feature flags
fourclass = True
Speaker_Disentanglement = True
Entropy = True
pretrain = True
pretraining = False
new_test = False

# Model paths
if pretrain:
    pathstr = r'/media/carol/Data/Documents/Emo_rec/NewMel/Pretraining/Domination'
    processor_path = os.path.join(pathstr, 'processor')
    model_path = os.path.join(pathstr, 'model')
elif model_type != 'ViT':
    model_path = 'shi-labs/dinat-mini-in1k-224'
    processor_path = 'shi-labs/dinat-mini-in1k-224'
    pathstr = model_path
else:
    model_path = 'google/vit-base-patch16-224-in21k'
    processor_path = 'google/vit-base-patch16-224-in21k'
    pathstr = model_path

# Emotion labels
EMOTIONS = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
}
Map2Num = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
}
Cval = 4

# Batch sizes
train_size = 40
eval_size = 40

# Training arguments
metric_name = "eval_uar"
args = TrainingArguments(
    f".././logs2",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    disable_tqdm=True,
    learning_rate=1e-5,
    lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
    warmup_ratio=0.1,
    per_device_train_batch_size=train_size,
    per_device_eval_batch_size=eval_size,
    num_train_epochs=10,
    weight_decay=wt_dc,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='.././logs_DO_NOT_DELETE/3090',
    remove_unused_columns=False,
)

# Grid search parameters
# Define grid search parameters
alphas = [0.5, 1.0, 1.5]
betas = [0.05, 0.08, 0.1, 1.0]
centers = [1.0, 1.6, 2.0]

# Class weights options (you can expand or modify these as needed)
class_weights_options = [
    [1.0, 1.0, 1.0, 1.0],  # Uniform weights
    [1.1, 1.1, 1.0, 1.0],  # Original weights
    [1.2, 1.2, 0.8, 1],  # Another option
]

# Set up results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join(r'/media/carol/Data/Documents/Emo_rec/GridSearch', timestamp)
os.makedirs(base_dir, exist_ok=True)

# Create CSV to store all results
results_csv = os.path.join(base_dir, 'grid_search_results.csv')
resume_file = os.path.join(base_dir, 'resume_state.json')

# Function to save the current state for resuming
def save_resume_state(completed_configurations):
    with open(resume_file, 'w') as f:
        json.dump(completed_configurations, f)

# Function to load the resume state
def load_resume_state():
    if os.path.exists(resume_file):
        with open(resume_file, 'r') as f:
            return json.load(f)
    return []

# Check if we need to resume from a previous run
completed_configurations = load_resume_state()

# Initialize results CSV if it doesn't exist
if not os.path.exists(results_csv):
    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ['run_id', 'alpha', 'beta', 'center', 'class_weights', 
                    'eval_acc', 'eval_uar', 'val_acc', 'val_uar']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

# Load datasets
dataset_train = 'cairocode/IEMO_Mel_6'
train_d0 = load_dataset(dataset_train, split='train')
train_d0 = train_d0.filter(filter_m_examples)

ds_eval_name = 'cairocode/MSPI_Mel6'
ds_eval = load_dataset(ds_eval_name, split = "train")
# Create a 80/20 split
train_val_split = train_d0.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Balance the train dataset if needed
train_dataset = balance_dataset(train_dataset, label_column="label", seed=42)

# Create list of all parameter combinations
param_combinations = list(itertools.product(alphas, betas, centers, range(len(class_weights_options))))

# Filter out already completed combinations
if completed_configurations:
    pending_combinations = [combo for combo in param_combinations 
                          if f"{combo[0]}_{combo[1]}_{combo[2]}_{combo[3]}" not in completed_configurations]
else:
    pending_combinations = param_combinations

print(f"Total combinations: {len(param_combinations)}")
print(f"Pending combinations: {len(pending_combinations)}")

# Run the grid search
for combo_idx, (alpha, beta, center, cw_idx) in enumerate(pending_combinations):
    run_id = f"{alpha}_{beta}_{center}_{cw_idx}"
    print(f"\n{'-'*80}")
    print(f"Starting run {combo_idx+1}/{len(pending_combinations)}: {run_id}")
    print(f"{'-'*80}")
    
    # Create unique directory for this run
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Get class weights for this run
    class_weights = class_weights_options[cw_idx]
    class_weights_tensor = torch.tensor(class_weights)
    
    try:
        # Initialize processor and model
        processor = AutoImageProcessor.from_pretrained(processor_path)
        
        base_model = DinatForImageClassification.from_pretrained(
            model_path,
            id2label=EMOTIONS,
            num_labels=Cval,
            label2id=Map2Num,
            ignore_mismatched_sizes=True,
            problem_type='single_label_classification',
            output_hidden_states=True
        )
        
        # Set up dataloader with speaker disentanglement if enabled
        if Speaker_Disentanglement:
            custom_dataset = CustomDataset(train_dataset)
            custom_sampler = CustomSampler(custom_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=custom_sampler, collate_fn=collate_fn, batch_size=train_size)
        else:
            train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=train_size)
        
        # Initialize the model with current hyperparameters
        model = CustomDinatForImageClassification_V2(
            base_model=base_model,
            num_classes=Cval,
            feature_dim=512,
            class_weights=class_weights_tensor,
            alpha=alpha,
            beta=beta,
            center_lr=center
        )
        
        model.to(device)
        
        # Super loss parameters
        super_loss_params = {
            'C': Cval,
            'lam': 0.01,
            'batch_size': args.train_batch_size,
            'class_weights': class_weights
        }
        
        # Set transforms
        val_dataset.set_transform(val_transforms)
        train_dataset.set_transform(train_transforms)
        ds_eval.set_transform(val_transforms)
        # Early stopping
        early_stopping = EarlyStoppingCallback(early_stopping_patience=12, early_stopping_threshold=0.001)
        
        # Train the model
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            tokenizer=processor,
            callbacks=[early_stopping],
        )
        
        trainer.train()
        
        # Save model and processor
        trainer.save_model(os.path.join(run_dir, "model"))
        processor.save_pretrained(os.path.join(run_dir, "processor"))
        
        # Evaluate on validation set
        val_outputs = trainer.predict(val_dataset)
        val_acc = val_outputs.metrics['test_accuracy'] * 100
        val_uar = val_outputs.metrics['test_uar'] * 100
        
        # Evaluate on training set
        eval_outputs = trainer.predict(ds_eval)
        eval_acc = eval_outputs.metrics['test_accuracy'] * 100
        eval_uar = eval_outputs.metrics['test_uar'] * 100
        
        # Save confusion matrices
        save_confusion_matrix(val_outputs, dataset_train, run_dir, Map2Num)
        save_confusion_matrix(eval_outputs, ds_eval_name, run_dir, Map2Num)
        
        # Save detailed model info
        model_info = {
            "Run ID": run_id,
            "Pretrain_file": pathstr,
            "Dataset Used": dataset_train,
            "Model Type": model_type,
            "Super Loss PARAMS": super_loss_params,
            "Speaker Disentanglement": Speaker_Disentanglement,
            "Entropy Curriculum Training": Entropy,
            "Test Results": val_outputs.metrics,
            "Eval Results": eval_outputs.metrics,
            "Class Weight": class_weights,
            "Weight decay": wt_dc,
            "alpha": alpha,
            "beta": beta,
            "center": center,
        }
        
        save_model_header(run_dir, model_info)
        
        # Append results to CSV
        with open(results_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['run_id', 'alpha', 'beta', 'center', 'class_weights', 
                                                      'eval_acc', 'eval_uar', 'val_acc', 'val_uar'])
            writer.writerow({
                'run_id': run_id,
                'alpha': alpha,
                'beta': beta,
                'center': center,
                'class_weights': str(class_weights),
                'xcorp_acc': eval_acc,
                'xcorp_uar': eval_uar,
                'val_acc': val_acc,
                'val_uar': val_uar
            })
        
        # Print results
        print(f"Run {run_id} completed")
        print(f"Xcorp Accuracy: {eval_acc:.2f}%, Xcorp UAR: {eval_uar:.2f}%")
        print(f"Val Accuracy: {val_acc:.2f}%, Val UAR: {val_uar:.2f}%")
        
        # Mark this configuration as completed
        completed_configurations.append(run_id)
        save_resume_state(completed_configurations)
        
        # Clean up to avoid memory issues
        torch.cuda.empty_cache()
        del trainer
        del model
        del base_model
        
    except Exception as e:
        print(f"Error in run {run_id}: {str(e)}")
        # Continue with next combination
        continue

# After completing all runs, find the best configuration
try:
    results_df = pd.read_csv(results_csv)
    
    # Find best by validation UAR (primary metric)
    best_val_uar_idx = results_df['val_uar'].idxmax()
    best_val_uar_config = results_df.iloc[best_val_uar_idx]
    
    # Find best by validation accuracy
    best_val_acc_idx = results_df['val_acc'].idxmax()
    best_val_acc_config = results_df.iloc[best_val_acc_idx]
    
    print("\n\n" + "="*50)
    print("GRID SEARCH COMPLETED")
    print("="*50)
    
    print("\nBest configuration by validation UAR:")
    print(f"Run ID: {best_val_uar_config['run_id']}")
    print(f"Alpha: {best_val_uar_config['alpha']}")
    print(f"Beta: {best_val_uar_config['beta']}")
    print(f"Center: {best_val_uar_config['center']}")
    print(f"Class Weights: {best_val_uar_config['class_weights']}")
    print(f"Validation UAR: {best_val_uar_config['val_uar']:.2f}%")
    print(f"Validation Accuracy: {best_val_uar_config['val_acc']:.2f}%")
    
    print("\nBest configuration by validation Accuracy:")
    print(f"Run ID: {best_val_acc_config['run_id']}")
    print(f"Alpha: {best_val_acc_config['alpha']}")
    print(f"Beta: {best_val_acc_config['beta']}")
    print(f"Center: {best_val_acc_config['center']}")
    print(f"Class Weights: {best_val_acc_config['class_weights']}")
    print(f"Validation Accuracy: {best_val_acc_config['val_acc']:.2f}%")
    print(f"Validation UAR: {best_val_acc_config['val_uar']:.2f}%")
    
    # Save summary to a text file
    with open(os.path.join(base_dir, 'grid_search_summary.txt'), 'w') as f:
        f.write("GRID SEARCH SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("Best configuration by validation UAR:\n")
        f.write(f"Run ID: {best_val_uar_config['run_id']}\n")
        f.write(f"Alpha: {best_val_uar_config['alpha']}\n")
        f.write(f"Beta: {best_val_uar_config['beta']}\n")
        f.write(f"Center: {best_val_uar_config['center']}\n")
        f.write(f"Class Weights: {best_val_uar_config['class_weights']}\n")
        f.write(f"Validation UAR: {best_val_uar_config['val_uar']:.2f}%\n")
        f.write(f"Validation Accuracy: {best_val_uar_config['val_acc']:.2f}%\n\n")
        
        f.write("Best configuration by validation Accuracy:\n")
        f.write(f"Run ID: {best_val_acc_config['run_id']}\n")
        f.write(f"Alpha: {best_val_acc_config['alpha']}\n")
        f.write(f"Beta: {best_val_acc_config['beta']}\n")
        f.write(f"Center: {best_val_acc_config['center']}\n")
        f.write(f"Class Weights: {best_val_acc_config['class_weights']}\n")
        f.write(f"Validation Accuracy: {best_val_acc_config['val_acc']:.2f}%\n")
        f.write(f"Validation UAR: {best_val_acc_config['val_uar']:.2f}%\n")
        
    # Create visualization of the results
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot heatmap of validation UAR for different alpha and beta values
        plt.figure(figsize=(12, 10))
        pivot_data = results_df.pivot_table(
            index='alpha', 
            columns='beta',
            values='val_uar',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Validation UAR by Alpha and Beta')
        plt.savefig(os.path.join(base_dir, 'alpha_beta_heatmap.png'))
        
        # Plot bar chart of top 10 configurations
        plt.figure(figsize=(15, 8))
        top10 = results_df.sort_values('val_uar', ascending=False).head(10)
        sns.barplot(x='run_id', y='val_uar', data=top10)
        plt.title('Top 10 Configurations by Validation UAR')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'top10_configurations.png'))
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        
except Exception as e:
    print(f"Error analyzing results: {str(e)}")

print("\nGrid search completed. All results saved to:", base_dir)