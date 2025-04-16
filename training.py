import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_one_epoch(
    model, 
    train_dataloader, 
    optimizer, 
    scheduler, 
    device, 
    epoch, 
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    log_interval=10,
    warmup_steps=0,
    warmup_ratio=0.1
):
    model.train()
    epoch_loss = 0
    step_loss = 0
    total_steps = len(train_dataloader)
    warmup_steps = int(warmup_ratio * total_steps) if warmup_steps == 0 else warmup_steps
    
    # Statistics tracking
    batch_times = []
    step_losses = []
    learning_rates = []
    
    start_time = time.time()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
    
    for step, batch in progress_bar:
        # Move batch to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        batch_start = time.time()
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Track loss
        step_loss += loss.item()
        
        # Update weights if gradient accumulation is complete
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate for warmup
            if step < warmup_steps:
                lr = optimizer.param_groups[0]['lr'] * ((step + 1) / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Log statistics
            if (step + 1) % log_interval == 0 or (step + 1) == len(train_dataloader):
                current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                
                # Calculate average loss over accumulation steps
                avg_loss = step_loss * gradient_accumulation_steps / log_interval
                step_losses.append(avg_loss)
                
                # Log to progress bar
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.6f}"
                })
                
                # Reset step loss
                step_loss = 0
        
        # Track batch time
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        
        # Update total loss
        epoch_loss += loss.item() * gradient_accumulation_steps
    
    # Update learning rate schedule at the end of epoch
    scheduler.step()
    
    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    
    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    total_epoch_time = time.time() - start_time
    
    # Return all statistics
    return {
        "epoch": epoch + 1,
        "avg_loss": avg_epoch_loss,
        "avg_batch_time": avg_batch_time,
        "total_time": total_epoch_time,
        "step_losses": step_losses,
        "learning_rates": learning_rates
    }

def evaluate(model, eval_dataloader, device, prefix="eval"):
    model.eval()
    all_preds = []
    all_logits = []
    all_labels = []
    total_loss = 0
    
    start_time = time.time()
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"{prefix.capitalize()}", leave=False):
            # Move batch to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs)
            
            # Collect predictions and labels
            logits = outputs.logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = inputs["labels"].cpu().numpy()
            
            all_preds.extend(preds)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels)
            total_loss += outputs.loss.item()
    
    # Concatenate logits from all batches
    all_logits = np.concatenate(all_logits, axis=0)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    uar = balanced_accuracy_score(all_labels, all_preds)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Total evaluation time
    eval_time = time.time() - start_time
    
    metrics = {
        f"{prefix}_loss": total_loss / len(eval_dataloader),
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_uar": uar,
        f"{prefix}_time": eval_time
    }
    
    return metrics, np.array(all_logits), np.array(all_labels), np.array(all_preds), cm

def predict(model, test_dataloader, device):
    metrics, logits, labels, preds, cm = evaluate(
        model=model,
        eval_dataloader=test_dataloader,
        device=device,
        prefix="test"
    )
    
    # Create a prediction result object similar to Trainer.predict output
    class PredictionOutput:
        def __init__(self, predictions, label_ids, metrics):
            self.predictions = predictions
            self.label_ids = label_ids
            self.metrics = metrics
            self.confusion_matrix = cm
    
    return PredictionOutput(logits, labels, metrics)

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "metrics": metrics
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer, scheduler, epoch, metrics

def plot_training_progress(training_stats, save_dir):
    """Plot training statistics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    epochs = [stat["epoch"] for stat in training_stats]
    train_losses = [stat["avg_loss"] for stat in training_stats]
    eval_losses = [stat.get("eval_loss", None) for stat in training_stats]
    eval_accuracies = [stat.get("eval_accuracy", None) * 100 if stat.get("eval_accuracy") else None for stat in training_stats]
    eval_uars = [stat.get("eval_uar", None) * 100 if stat.get("eval_uar") else None for stat in training_stats]
    
    # Remove None values
    eval_epochs = [epochs[i] for i in range(len(epochs)) if eval_losses[i] is not None]
    eval_losses = [l for l in eval_losses if l is not None]
    eval_accuracies = [a for a in eval_accuracies if a is not None]
    eval_uars = [u for u in eval_uars if u is not None]
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    if eval_losses:
        plt.plot(eval_epochs, eval_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    
    if eval_accuracies:
        plt.plot(eval_epochs, eval_accuracies, 'g-', label='Accuracy (%)')
    
    if eval_uars:
        plt.plot(eval_epochs, eval_uars, 'c-', label='UAR (%)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value (%)')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_progress.png"))
    plt.close()
    
    # Plot learning rates
    learning_rates = []
    for stat in training_stats:
        if "learning_rates" in stat:
            # Add epoch information to each learning rate
            for i, lr in enumerate(stat["learning_rates"]):
                step = (stat["epoch"] - 1) + (i / len(stat["learning_rates"]))
                learning_rates.append((step, lr))
    
    if learning_rates:
        steps, lrs = zip(*learning_rates)
        
        plt.figure(figsize=(10, 4))
        plt.plot(steps, lrs, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "learning_rate.png"))
        plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate(
    model, 
    train_dataset, 
    eval_dataset, 
    test_dataset,
    collate_fn,
    num_epochs=50,
    learning_rate=1e-5,
    weight_decay=0.05,
    train_batch_size=20,
    eval_batch_size=20,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    device="cuda",
    early_stopping_patience=12,
    early_stopping_threshold=0.001,
    output_dir=None,
    class_names=None,
    resume_from_checkpoint=None,
    evaluation_strategy="epoch",  # "epoch", "steps", or "no"
    eval_steps=None,  # Evaluate every N steps if evaluation_strategy="steps"
    save_strategy="epoch",  # "epoch", "steps", or "no"
    save_steps=None,  # Save checkpoint every N steps if save_strategy="steps"
    save_total_limit=3,  # Maximum number of checkpoints to keep
    use_speaker_disentanglement=False,
    logging_dir=None
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
    
    # Create dataloaders
    if use_speaker_disentanglement:
        custom_dataset = CustomDataset(train_dataset)
        custom_sampler = CustomSampler(custom_dataset)
        train_dataloader = DataLoader(
            train_dataset, 
            sampler=custom_sampler, 
            collate_fn=collate_fn, 
            batch_size=train_batch_size
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            collate_fn=collate_fn, 
            shuffle=True
        )
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=eval_batch_size, 
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size, 
        collate_fn=collate_fn
    )
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Setup scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Restart after T_0 epochs
        T_mult=2,  # Multiply T_0 by this factor after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Training state
    start_epoch = 0
    global_step = 0
    training_stats = []
    best_metric = -float("inf")
    best_model_state = None
    patience_counter = 0
    checkpoints = []
    
    # Resume from checkpoint if provided
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        model, optimizer, scheduler, start_epoch, checkpoint_metrics = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=resume_from_checkpoint,
            device=device
        )
        
        # Update best metric if available
        if "eval_uar" in checkpoint_metrics:
            best_metric = checkpoint_metrics["eval_uar"]
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train for one epoch
        epoch_stats = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio
        )
        
        global_step += len(train_dataloader)
        
        # Evaluate if needed
        should_evaluate = (
            (evaluation_strategy == "epoch") or
            (evaluation_strategy == "steps" and global_step % eval_steps == 0)
        )
        
        if should_evaluate:
            eval_results, _, _, _, _ = evaluate(
                model=model,
                eval_dataloader=eval_dataloader,
                device=device
            )
            
            # Update epoch stats with evaluation results
            epoch_stats.update(eval_results)
            
            # Print metrics
            print(f"Train Loss: {epoch_stats['avg_loss']:.4f}, "
                  f"Eval Loss: {eval_results['eval_loss']:.4f}, "
                  f"Eval Accuracy: {eval_results['eval_accuracy']*100:.2f}%, "
                  f"Eval UAR: {eval_results['eval_uar']*100:.2f}%")
            
            # Check for improvement
            current_metric = eval_results["eval_uar"]
            if current_metric > best_metric + early_stopping_threshold:
                print(f"Improvement found! {current_metric:.4f} > {best_metric:.4f}")
                best_metric = current_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break
        
        # Save checkpoint if needed
        should_save = (
            (save_strategy == "epoch") or
            (save_strategy == "steps" and global_step % save_steps == 0)
        )
        
        if should_save and output_dir:
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics=epoch_stats,
                save_path=os.path.join(checkpoint_path, "training_state.pt")
            )
            
            # Also save model separately for easy loading
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
            
            # Keep track of checkpoints and remove old ones if needed
            checkpoints.append(checkpoint_path)
            if save_total_limit and len(checkpoints) > save_total_limit:
                # Remove oldest checkpoint
                oldest_checkpoint = checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    print(f"Removing old checkpoint: {oldest_checkpoint}")
                    import shutil
                    shutil.rmtree(oldest_checkpoint)
        
        # Save statistics
        training_stats.append(epoch_stats)
        
        # Save training progress plots
        if output_dir:
            plot_training_progress(training_stats, output_dir)
            
            # Save training stats as JSON
            with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
                json.dump(training_stats, f, indent=2)
    
    # End of training - load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with eval_uar: {best_metric:.4f}")
        
    # Save final model
    if output_dir:
        final_model_path = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(final_model_path, "pytorch_model.bin"))
        print(f"Final model saved to {final_model_path}")
    
    # Final evaluation on test dataset
    print("\nRunning final evaluation on test dataset...")
    test_results = predict(
        model=model,
        test_dataloader=test_dataloader,
        device=device
    )
    
    print(f"Test Accuracy: {test_results.metrics['test_accuracy']*100:.2f}%, "
          f"Test UAR: {test_results.metrics['test_uar']*100:.2f}%")
    
    # Save confusion matrix
    if output_dir and class_names:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(test_results.confusion_matrix, class_names, cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
        # Save test metrics
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(test_results.metrics, f, indent=2)
    
    return test_results, model