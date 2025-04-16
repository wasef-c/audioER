import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import http.server
import socketserver
import webbrowser
import logging
from pathlib import Path
import datetime
import shutil
import pandas as pd
import gc

class RobustHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler that gracefully handles broken pipes and client disconnects"""
    
    def log_message(self, format, *args):
        # Override to suppress logging
        pass
    
    def copyfile(self, source, outputfile):
        """Copy a file object to the outputfile, handling broken pipes gracefully"""
        try:
            shutil.copyfileobj(source, outputfile)
        except BrokenPipeError:
            # Client disconnected, close the connection gracefully
            self.connection.close()
            return
        except ConnectionResetError:
            # Connection reset by peer
            self.connection.close()
            return
        except Exception as e:
            # Log other exceptions but don't crash
            print(f"Error serving file: {str(e)}")
            self.connection.close()
            return

class MultiSpeakerDashboard:
    """
    An enhanced dashboard that supports tracking multiple sequential training runs,
    perfect for speaker-by-speaker training loops.
    """
    def __init__(self, base_dir, port=8000, auto_open=True, max_batch_records = 50,plot_dpi = 100, auto_refresh_interval = 10 ):
        self.base_dir = Path(base_dir)
        self.dashboard_dir = self.base_dir / 'dashboard'
        self.dashboard_dir.mkdir(exist_ok=True)
        
        # Create history directory to store previous runs
        self.history_dir = self.dashboard_dir / 'history'
        self.history_dir.mkdir(exist_ok=True)
        
        self.current_metrics_file = self.dashboard_dir / 'current_metrics.json'
        self.batch_metrics_file = self.dashboard_dir / 'batch_metrics.json'
        self.plots_dir = self.dashboard_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        self.port = port
        self.server = None
        self.server_thread = None
        self.auto_open = auto_open

        # Rate limiting for updates
        self.last_plot_update = 0
        self.plot_update_interval = 2.0  # seconds
        self.last_file_write = 0
        self.file_write_interval = 0.5  # seconds
        # Configuration options
        self.max_batch_records = max_batch_records  # Default to 50
        self.plot_dpi = plot_dpi  # Default to 100
        self.auto_refresh_interval = auto_refresh_interval  # Default to 10
        
        # Current run information
        self.current_speaker = None
        self.current_speaker_id = None
        self.speaker_history = []
        
        # Batch tracking data
        self.batch_x_values = []
        self.batch_y_values = []
        
        # Initialize metrics storage
        self.reset_metrics()
        
        # Create the HTML dashboard
        self._create_dashboard_html()

        
        
        # Start the server
        if auto_open:
            self.start_server()
    
    def reset_metrics(self):
        """Reset metrics for a new training run"""
        self.metrics = {
            'speaker': self.current_speaker,
            'speaker_id': self.current_speaker_id,
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'accuracies': [],
            'uars': [],
            'f1_scores': [],
            'last_update': None
        }
        
        # Also reset batch metrics
        self.batch_metrics = {
            'speaker': self.current_speaker,
            'speaker_id': self.current_speaker_id,
            'current_epoch': 0,
            'current_batch': 0,
            'total_batches': 0,
            'batch_losses': [],
            'last_update': None
        }
    
    def start_speaker_run(self, speaker_id, speaker_name=None):
        """
        Start tracking a new speaker's training run.
        Archives the previous run's data if it exists.
        
        Args:
            speaker_id: Unique identifier for the speaker
            speaker_name: Descriptive name for the speaker (optional)
        """
        # If we had a previous run, archive it
        if self.current_speaker_id is not None:
            self._archive_current_run()
        
        # Set new speaker info
        self.current_speaker_id = speaker_id
        self.current_speaker = speaker_name or f"Speaker {speaker_id}"
        
        # Reset metrics for new run
        self.reset_metrics()
        
        # Reset batch tracking data
        self.batch_x_values = []
        self.batch_y_values = []
        
        # Update metrics files with new speaker info
        self.metrics['speaker'] = self.current_speaker
        self.metrics['speaker_id'] = self.current_speaker_id
        self.batch_metrics['speaker'] = self.current_speaker
        self.batch_metrics['speaker_id'] = self.current_speaker_id
        
        # Save initial metrics files
        with open(self.current_metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        with open(self.batch_metrics_file, 'w') as f:
            json.dump(self.batch_metrics, f, indent=2)
        
        # Update speaker history file
        self._update_speaker_history()
        
        # Generate an initial empty plot
        self._generate_plots()
        
        print(f"Started tracking training for {self.current_speaker} (ID: {self.current_speaker_id})")
    
    def _archive_current_run(self):
        """Archive the current run's data to history"""
        if self.current_speaker_id is None:
            return
        
        # Create a directory for this speaker's history
        speaker_dir = self.history_dir / f"speaker_{self.current_speaker_id}"
        speaker_dir.mkdir(exist_ok=True)
        
        # Save the current metrics to the speaker's directory
        with open(speaker_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save the current plot to the speaker's directory
        if (self.plots_dir / "training_progress.png").exists():
            shutil.copy(
                self.plots_dir / "training_progress.png", 
                speaker_dir / f"training_progress.png"
            )
        
        # Add to speaker history
        if self.current_speaker_id not in [s['speaker_id'] for s in self.speaker_history]:
            self.speaker_history.append({
                'speaker_id': self.current_speaker_id,
                'speaker_name': self.current_speaker,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'metrics': {
                    'final_accuracy': self.metrics['accuracies'][-1] if self.metrics['accuracies'] else None,
                    'final_uar': self.metrics['uars'][-1] if self.metrics['uars'] else None,
                    'final_f1': self.metrics['f1_scores'][-1] if self.metrics['f1_scores'] else None,
                    'best_accuracy': max(self.metrics['accuracies']) if self.metrics['accuracies'] else None,
                    'best_uar': max(self.metrics['uars']) if self.metrics['uars'] else None,
                    'best_f1': max(self.metrics['f1_scores']) if self.metrics['f1_scores'] else None,
                    'epochs_trained': len(self.metrics['epochs'])
                }
            })
    
    def _update_speaker_history(self):
        """Update the speaker history file"""
        history_file = self.dashboard_dir / "speaker_history.json"
        
        with open(history_file, "w") as f:
            json.dump(self.speaker_history, f, indent=2)
    
    def update(self, epoch, train_loss, val_loss=None, accuracy=None, uar=None, f1=None, save_plots=True):
        """
        Update the dashboard with new metrics.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            accuracy: Validation accuracy (optional)
            uar: Validation UAR (optional)
            f1: Validation F1 score (optional)
            save_plots: Whether to save plots (default: True)
        """
        # Skip if no active speaker
        if self.current_speaker_id is None:
            print("Warning: No active speaker session. Call start_speaker_run first.")
            return
        
        # Update metrics
        self.metrics['epochs'].append(epoch)
        self.metrics['train_losses'].append(float(train_loss))
        
        # For validation metrics, use the last value if not provided
        if val_loss is not None:
            self.metrics['val_losses'].append(float(val_loss))
        elif self.metrics['val_losses']:
            self.metrics['val_losses'].append(self.metrics['val_losses'][-1])
        else:
            self.metrics['val_losses'].append(None)
            
        if accuracy is not None:
            self.metrics['accuracies'].append(float(accuracy))
        elif self.metrics['accuracies']:
            self.metrics['accuracies'].append(self.metrics['accuracies'][-1])
        else:
            self.metrics['accuracies'].append(None)
            
        if uar is not None:
            self.metrics['uars'].append(float(uar))
        elif self.metrics['uars']:
            self.metrics['uars'].append(self.metrics['uars'][-1])
        else:
            self.metrics['uars'].append(None)
            
        if f1 is not None:
            self.metrics['f1_scores'].append(float(f1))
        elif self.metrics['f1_scores']:
            self.metrics['f1_scores'].append(self.metrics['f1_scores'][-1])
        else:
            self.metrics['f1_scores'].append(None)
        
        # Update timestamp
        self.metrics['last_update'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save metrics to file
        with open(self.current_metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Generate and save plots if requested
        if save_plots:
            self._generate_plots()
    
    def update_batch(self, epoch, batch, train_loss, total_batches):
        """
        Update training metrics for a specific batch within an epoch.
        Records batch loss for real-time plotting.
        
        Args:
            epoch: Current epoch number
            batch: Current batch number
            train_loss: Current batch training loss
            total_batches: Total number of batches per epoch
        """
        # Skip if no active speaker
        if self.current_speaker_id is None:
            return
            
        # Update batch metrics
        self.batch_metrics['current_epoch'] = epoch
        self.batch_metrics['current_batch'] = batch
        self.batch_metrics['total_batches'] = total_batches
        self.batch_metrics['batch_losses'].append(float(train_loss))
        self.batch_metrics['last_update'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Append to batch losses, maintaining max size
        self.batch_metrics["batch_losses"].append(float(train_loss))
        if len(self.batch_metrics["batch_losses"]) > self.max_batch_records:
            self.batch_metrics["batch_losses"] = self.batch_metrics["batch_losses"][-self.max_batch_records:]

        # Keep arrays in sync with stored batch losses
        if len(self.batch_x_values) > self.max_batch_records:
            self.batch_x_values = self.batch_x_values[-self.max_batch_records:]
            self.batch_y_values = self.batch_y_values[-self.max_batch_records:]
        
        # Calculate global batch number (epoch * total_batches + batch)
        # This ensures x values properly represent progress across epochs
        global_batch_num = (epoch * total_batches) + batch
        
        # Track batch data for plotting
        self.batch_x_values.append(global_batch_num)
        self.batch_y_values.append(float(train_loss))
        
        # Keep arrays in sync with stored batch losses
        if len(self.batch_metrics['batch_losses']) > 100:
            self.batch_metrics['batch_losses'] = self.batch_metrics['batch_losses'][-100:]
            self.batch_x_values = self.batch_x_values[-100:]
            self.batch_y_values = self.batch_y_values[-100:]
        
        # Save batch metrics to file
        with open(self.batch_metrics_file, 'w') as f:
            json.dump(self.batch_metrics, f, indent=2)
        
        # Update plots
        self._generate_plots()

    def _write_metrics_files(self, force=False):
        """Write metrics to files with rate limiting to reduce I/O"""
        current_time = time.time()
        
        # Skip if we wrote recently unless force=True
        if not force and (current_time - self.last_file_write) < self.file_write_interval:
            return
            
        # Update last write time
        self.last_file_write = current_time
        
        # Write files with simplified JSON (no indentation)
        with open(self.current_metrics_file, "w") as f:
            json.dump(self.metrics, f)  # Removed indent for smaller files

    def _generate_plots(self, force=False):
        """Generate plots with rate limiting"""
        current_time = time.time()
        
        # Skip if we updated recently unless force=True
        if not force and (current_time - self.last_plot_update) < self.plot_update_interval:
            return
        """Generate plots for the dashboard with improved batch-level data visualization"""
        # Skip if no data
        if not self.batch_x_values and not self.metrics['epochs']:
            # Create an empty plot as placeholder
            plt.figure(figsize=(12, 10))
            plt.figtext(0.5, 0.5, f"No data yet for {self.current_speaker}",
                    ha='center', va='center', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'training_progress.png')
            plt.close()
            return
                
        # Set up the figure - 2x2 grid
        plt.figure(figsize=(12, 10))
        
        # Add speaker info at the top
        plt.suptitle(f"Training Progress for {self.current_speaker} (ID: {self.current_speaker_id})", 
                    fontsize=16, y=0.98)
        
        # Find best metrics and their indices
        best_accuracy, best_accuracy_idx = None, None
        best_uar, best_uar_idx = None, None
        best_f1, best_f1_idx = None, None
        
        if self.metrics['accuracies'] and any(a is not None for a in self.metrics['accuracies']):
            valid_accuracies = [a for a in self.metrics['accuracies'] if a is not None]
            if valid_accuracies:
                best_accuracy = max(valid_accuracies)
                best_accuracy_idx = self.metrics['accuracies'].index(best_accuracy)
        
        if self.metrics['uars'] and any(u is not None for u in self.metrics['uars']):
            valid_uars = [u for u in self.metrics['uars'] if u is not None]
            if valid_uars:
                best_uar = max(valid_uars)
                best_uar_idx = self.metrics['uars'].index(best_uar)
        
        if self.metrics['f1_scores'] and any(f is not None for f in self.metrics['f1_scores']):
            valid_f1s = [f for f in self.metrics['f1_scores'] if f is not None]
            if valid_f1s:
                best_f1 = max(valid_f1s)
                best_f1_idx = self.metrics['f1_scores'].index(best_f1)
        
        # Plot 1: Training Loss (both epoch and batch level)
        plt.subplot(2, 2, 1)
        
        # Plot batch-level training loss if available
        if self.batch_x_values:
            # Convert to numpy arrays for easier manipulation
            x = np.array(self.batch_x_values)
            y = np.array(self.batch_y_values)
            
            # Calculate total batches per epoch for scaling
            total_batches = self.batch_metrics['total_batches'] or 1
            
            # Scale x to represent epochs + batch progress
            if total_batches > 0:
                # Convert global batch numbers to epoch format
                x_scaled = x / total_batches
                
                # IMPROVED: Group batch data by epoch for better visualization
                max_epoch = int(np.max(x_scaled)) + 1
                
                # Get current epoch for determining visualization strategy
                current_epoch = self.batch_metrics['current_epoch']
                
                # IMPROVED: Apply intelligent downsampling for large datasets while preserving all epochs
                if len(x) > 10000:  # If we have a lot of points
                    # Define maximum points to plot per epoch to avoid performance issues
                    max_points_per_epoch = 200
                    
                    # Create empty arrays for downsampled data
                    x_downsampled = []
                    y_downsampled = []
                    
                    # Process each epoch separately
                    for epoch in range(max_epoch + 1):
                        # Get indices for this epoch
                        epoch_indices = np.where((x_scaled >= epoch) & (x_scaled < epoch + 1))[0]
                        
                        if len(epoch_indices) > 0:
                            # If we have more points than our limit, downsample
                            if len(epoch_indices) > max_points_per_epoch:
                                # Use systematic sampling to preserve pattern
                                sample_indices = np.linspace(0, len(epoch_indices) - 1, max_points_per_epoch, dtype=int)
                                downsampled_indices = epoch_indices[sample_indices]
                                
                                # Add to our downsampled arrays
                                x_downsampled.extend(x_scaled[downsampled_indices])
                                y_downsampled.extend(y[downsampled_indices])
                            else:
                                # Use all points for this epoch
                                x_downsampled.extend(x_scaled[epoch_indices])
                                y_downsampled.extend(y[epoch_indices])
                    
                    # Use downsampled data for plotting
                    plt.plot(x_downsampled, y_downsampled, 'b-', alpha=0.5, linewidth=1, label='Batch Loss')
                else:
                    # If dataset is small enough, just plot all points
                    plt.plot(x_scaled, y, 'b-', alpha=0.5, linewidth=1, label='Batch Loss')
                
                # Add dots for epoch markers if we have epoch data
                if self.metrics['epochs'] and self.metrics['train_losses']:
                    plt.plot(self.metrics['epochs'], self.metrics['train_losses'], 'bo', markersize=6, label='Epoch Average')
            
            # Set title and labels
            plt.title('Training Loss (Batch Level)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Ensure x-axis extends to at least cover all completed epochs
            if current_epoch > 0:  # Only if we're past epoch 0
                plt.xlim(0, max(current_epoch + 0.5, max(x_scaled) + 0.1))
        
        # Plot 2: Validation Loss (epoch level only)
        plt.subplot(2, 2, 2)
        if self.metrics['epochs'] and any(x is not None for x in self.metrics['val_losses']):
            plt.plot(self.metrics['epochs'], self.metrics['val_losses'], 'r-', label='Validation Loss')
            plt.title('Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.title('Validation Loss (No Data Yet)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Validation Accuracy with best point highlighted
        plt.subplot(2, 2, 3)
        if self.metrics['epochs'] and any(x is not None for x in self.metrics['accuracies']):
            plt.plot(self.metrics['epochs'], self.metrics['accuracies'], 'g-', label='Accuracy')
            
            # Highlight best accuracy point if available
            if best_accuracy is not None and best_accuracy_idx is not None:
                plt.plot(self.metrics['epochs'][best_accuracy_idx], best_accuracy, 'g*', 
                        markersize=12, label=f'Best: {best_accuracy:.4f} (Epoch {self.metrics["epochs"][best_accuracy_idx]})')
            
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.title('Validation Accuracy (No Data Yet)')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Validation UAR and F1 Score with best points highlighted
        plt.subplot(2, 2, 4)
        if self.metrics['epochs']:
            if any(x is not None for x in self.metrics['uars']):
                plt.plot(self.metrics['epochs'], self.metrics['uars'], 'm-', label='UAR')
                
                # Highlight best UAR point if available
                if best_uar is not None and best_uar_idx is not None:
                    plt.plot(self.metrics['epochs'][best_uar_idx], best_uar, 'm*', 
                            markersize=12, label=f'Best UAR: {best_uar:.4f} (Epoch {self.metrics["epochs"][best_uar_idx]})')
            
            if any(x is not None for x in self.metrics['f1_scores']):
                plt.plot(self.metrics['epochs'], self.metrics['f1_scores'], 'c-', label='F1 Score')
                
                # Highlight best F1 point if available
                if best_f1 is not None and best_f1_idx is not None:
                    plt.plot(self.metrics['epochs'][best_f1_idx], best_f1, 'c*', 
                            markersize=12, label=f'Best F1: {best_f1:.4f} (Epoch {self.metrics["epochs"][best_f1_idx]})')
            
            plt.title('UAR and F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.title('UAR and F1 Score (No Data Yet)')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(self.plots_dir / 'training_progress.png')
        # Close all figures and force garbage collection
        plt.close('all')
        gc.collect()
        self.last_plot_update = current_time

    def start_server(self):
        """Start a robust HTTP server to serve the dashboard"""
        # Suppress HTTP server logs
        logging.getLogger('http.server').setLevel(logging.CRITICAL)
        
        # Start server in a separate thread
        self.server = socketserver.TCPServer(("", self.port), RobustHTTPRequestHandler)
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Change directory to dashboard_dir
        os.chdir(self.dashboard_dir)
        
        print(f"Dashboard server started at http://localhost:{self.port}/dashboard.html")
        
        # Open browser automatically if requested
        if self.auto_open:
            webbrowser.open(f"http://localhost:{self.port}/dashboard.html")
    
    def _run_server(self):
        """Run the HTTP server"""
        try:
            self.server.serve_forever()
        except (KeyboardInterrupt, SystemExit):
            self.server.shutdown()
    
    def stop_server(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server_thread.join()
            print("Dashboard server stopped")



    def _create_dashboard_html(self):
        """Create the HTML dashboard"""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Multi-Speaker Training Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                .speaker-info {
                    text-align: center;
                    margin-bottom: 20px;
                    padding: 10px;
                    background-color: #f0f8ff;
                    border-radius: 8px;
                    border-left: 5px solid #4169e1;
                }
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .metric-card {
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                }
                .metric-title {
                    margin-top: 0;
                    color: #555;
                    font-size: 1rem;
                }
                .metric-value {
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #333;
                }
                .metric-epoch {
                    font-size: 0.9rem;
                    color: #666;
                    margin-top: 5px;
                }
                .highlight {
                    background-color: #e6f7ff;
                    border-left: 3px solid #1890ff;
                }
                .plots {
                    text-align: center;
                    margin-top: 30px;
                }
                .plots img {
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .batch-progress {
                    margin-top: 20px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    padding: 15px;
                }
                .progress-bar {
                    width: 100%;
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    overflow: hidden;
                }
                .progress-fill {
                    height: 24px;
                    background-color: #4CAF50;
                    border-radius: 5px;
                    text-align: center;
                    line-height: 24px;
                    color: white;
                    transition: width 0.5s;
                }
                .refresh-info {
                    text-align: center;
                    margin-top: 20px;
                    color: #777;
                }
                .updated-at {
                    text-align: right;
                    margin-top: 20px;
                    font-style: italic;
                    color: #777;
                }
                .history-section {
                    margin-top: 40px;
                    border-top: 1px solid #ddd;
                    padding-top: 20px;
                }
                .history-section h2 {
                    color: #333;
                    margin-bottom: 15px;
                }
                .history-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }
                .history-table th, .history-table td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                .history-table th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                .history-table tr:hover {
                    background-color: #f5f5f5;
                }
                .view-button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 14px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 4px;
                }
                .modal {
                    display: none;
                    position: fixed;
                    z-index: 1;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    overflow: auto;
                    background-color: rgba(0,0,0,0.4);
                }
                .modal-content {
                    background-color: #fefefe;
                    margin: 5% auto;
                    padding: 20px;
                    border: 1px solid #888;
                    width: 80%;
                    max-width: 800px;
                    border-radius: 8px;
                }
                .close-button {
                    color: #aaa;
                    float: right;
                    font-size: 28px;
                    font-weight: bold;
                    cursor: pointer;
                }
                .close-button:hover {
                    color: black;
                }
                .modal-image {
                    width: 100%;
                    margin-top: 20px;
                    border-radius: 8px;
                }
                .tabs {
                    display: flex;
                    margin-bottom: 20px;
                }
                .tab {
                    padding: 10px 20px;
                    cursor: pointer;
                    background-color: #f2f2f2;
                    border: 1px solid #ddd;
                    border-bottom: none;
                    border-radius: 5px 5px 0 0;
                    margin-right: 5px;
                }
                .tab.active {
                    background-color: white;
                    border-bottom: 2px solid white;
                }
                .tab-content {
                    display: none;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 0 5px 5px 5px;
                }
                .tab-content.active {
                    display: block;
                }
                .best-metrics {
                    margin: 20px 0;
                    background-color: #f0f8ff;
                    border-radius: 8px;
                    padding: 15px;
                    border-left: 5px solid #1e90ff;
                }
                .best-metrics h3 {
                    margin-top: 0;
                    color: #333;
                }
                .best-metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Multi-Speaker Training Dashboard</h1>
                
                <div class="tabs">
                    <div id="current-tab" class="tab active" onclick="showTab('current-content')">Current Speaker</div>
                    <div id="history-tab" class="tab" onclick="showTab('history-content')">Speaker History</div>
                </div>
                
                <div id="current-content" class="tab-content active">
                    <div id="speaker-info" class="speaker-info">
                        <h2>Current Speaker: <span id="current-speaker">None</span></h2>
                        <div id="speaker-id">Speaker ID: None</div>
                    </div>
                    <div class="controls">
                    <div>
                        <button id="refresh-button" class="control-button">Refresh Now</button>
                        <button id="toggle-auto-refresh" class="control-button">Pause Auto-Refresh</button>
                    </div>
                    <div>
                        <select id="refresh-interval" title="Auto-refresh interval">
                            <option value="5">5 seconds</option>
                            <option value="10" selected>10 seconds</option>
                            <option value="30">30 seconds</option>
                            <option value="60">1 minute</option>
                            <option value="300">5 minutes</option>
                        </select>
                    </div>
                </div>
                    <div class="batch-progress">
                        <h3>Current Progress</h3>
                        <div id="epoch-info">Epoch: 0 / ?</div>
                        <div id="batch-info">Batch: 0 / 0</div>
                        <div class="progress-bar">
                            <div id="progress-fill" class="progress-fill" style="width: 0%">0%</div>
                        </div>
                    </div>
                    
                    <!-- Best metrics section -->
                    <div class="best-metrics">
                        <h3>Best Performance Metrics</h3>
                        <div class="best-metrics-grid">
                            <div class="metric-card highlight">
                                <h4 class="metric-title">Best Accuracy</h4>
                                <div id="best-accuracy" class="metric-value">N/A</div>
                                <div id="best-accuracy-epoch" class="metric-epoch">Epoch: N/A</div>
                            </div>
                            <div class="metric-card highlight">
                                <h4 class="metric-title">Best UAR</h4>
                                <div id="best-uar" class="metric-value">N/A</div>
                                <div id="best-uar-epoch" class="metric-epoch">Epoch: N/A</div>
                            </div>
                            <div class="metric-card highlight">
                                <h4 class="metric-title">Best F1 Score</h4>
                                <div id="best-f1" class="metric-value">N/A</div>
                                <div id="best-f1-epoch" class="metric-epoch">Epoch: N/A</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric-card">
                            <h3 class="metric-title">Latest Train Loss</h3>
                            <div id="train-loss" class="metric-value">N/A</div>
                        </div>
                        <div class="metric-card">
                            <h3 class="metric-title">Latest Validation Loss</h3>
                            <div id="val-loss" class="metric-value">N/A</div>
                        </div>
                        <div class="metric-card">
                            <h3 class="metric-title">Latest Accuracy</h3>
                            <div id="accuracy" class="metric-value">N/A</div>
                        </div>
                        <div class="metric-card">
                            <h3 class="metric-title">Latest UAR</h3>
                            <div id="uar" class="metric-value">N/A</div>
                        </div>
                        <div class="metric-card">
                            <h3 class="metric-title">Latest F1 Score</h3>
                            <div id="f1" class="metric-value">N/A</div>
                        </div>
                    </div>
                    
                    <div class="plots">
                        <h3>Training Progress</h3>
                        <img id="plots-img" src="plots/training_progress.png" alt="Training Progress" onerror="this.src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='">
                    </div>
                    
                    <div id="updated-at" class="updated-at">
                        Last updated: Never
                    </div>
                </div>
                
                <div id="history-content" class="tab-content">
                    <div class="history-section">
                        <h2>Speaker Training History</h2>
                        <table class="history-table">
                            <thead>
                                <tr>
                                    <th>Speaker</th>
                                    <th>ID</th>
                                    <th>Timestamp</th>
                                    <th>Best Accuracy</th>
                                    <th>Best UAR</th>
                                    <th>Epochs</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="history-tbody">
                                <!-- History rows will be added here -->
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="refresh-info">
                    <p>This page auto-refreshes every 5 seconds.</p>
                </div>
            </div>
            
            <!-- Modal for viewing historical data -->
            <div id="history-modal" class="modal">
                <div class="modal-content">
                    <span class="close-button" onclick="closeModal()">&times;</span>
                    <h2 id="modal-title">Speaker History</h2>
                    <div id="modal-content">
                        <!-- Modal content will be added here -->
                    </div>
                    <img id="modal-image" class="modal-image" src="" alt="Historical Training Progress">
                </div>
            </div>
            
            <script>
                let autoRefreshEnabled = true;
                let refreshIntervalId = null;
                let refreshInterval = 10; // seconds (default refresh interval)
                let lastRefreshTimestamp = 0;
                let pendingRefresh = false;

                // Tab switching functionality
                function showTab(tabId) {
                    // Hide all tab contents
                    document.querySelectorAll('.tab-content').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    
                    // Remove active class from all tabs
                    document.querySelectorAll('.tab').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    
                    // Show the selected tab content and activate the tab
                    document.getElementById(tabId).classList.add('active');
                    
                    if (tabId === 'current-content') {
                        document.getElementById('current-tab').classList.add('active');
                    } else if (tabId === 'history-content') {
                        document.getElementById('history-tab').classList.add('active');
                        loadSpeakerHistory();
                    }
                }
                
                // Function to load speaker history
                function loadSpeakerHistory() {
                    const timestamp = new Date().getTime();
                    fetch('speaker_history.json?t=' + timestamp)
                        .then(response => response.json())
                        .then(data => {
                            const historyTable = document.getElementById('history-tbody');
                            historyTable.innerHTML = '';
                            
                            // Sort history by timestamp (newest first)
                            data.sort((a, b) => {
                                return new Date(b.timestamp) - new Date(a.timestamp);
                            });
                            
                            data.forEach(speaker => {
                                const row = document.createElement('tr');
                                
                                // Format metrics to 2 decimal places or show N/A
                                const formatMetric = (value) => {
                                    return value !== null ? (value * 100).toFixed(2) + '%' : 'N/A';
                                };
                                
                                row.innerHTML = `
                                    <td>${speaker.speaker_name}</td>
                                    <td>${speaker.speaker_id}</td>
                                    <td>${speaker.timestamp}</td>
                                    <td>${formatMetric(speaker.metrics.best_accuracy)}</td>
                                    <td>${formatMetric(speaker.metrics.best_uar)}</td>
                                    <td>${speaker.metrics.epochs_trained}</td>
                                    <td>
                                        <button class="view-button" onclick="viewSpeakerHistory(${speaker.speaker_id})">
                                            View
                                        </button>
                                    </td>
                                `;
                                
                                historyTable.appendChild(row);
                            });
                        })
                        .catch(error => {
                            console.error('Error loading speaker history:', error);
                        });
                }
                
                // Function to view a specific speaker's history
                function viewSpeakerHistory(speakerId) {
                    const modal = document.getElementById('history-modal');
                    const modalTitle = document.getElementById('modal-title');
                    const modalContent = document.getElementById('modal-content');
                    const modalImage = document.getElementById('modal-image');
                    
                    // Load the speaker's metrics
                    fetch(`history/speaker_${speakerId}/metrics.json`)
                        .then(response => response.json())
                        .then(data => {
                            modalTitle.textContent = `History for ${data.speaker} (ID: ${data.speaker_id})`;
                            
                            // Create a summary of the metrics
                            const lastIdx = data.epochs.length - 1;
                            const bestAccIdx = data.accuracies.indexOf(Math.max(...data.accuracies));
                            const bestUarIdx = data.uars.indexOf(Math.max(...data.uars));
                            
                            modalContent.innerHTML = `
                                <div style="margin-bottom: 20px;">
                                    <h3>Training Summary</h3>
                                    <p><strong>Total Epochs:</strong> ${data.epochs.length}</p>
                                    <p><strong>Final Metrics:</strong> Accuracy: ${(data.accuracies[lastIdx] * 100).toFixed(2)}%, 
                                    UAR: ${(data.uars[lastIdx] * 100).toFixed(2)}%, 
                                    F1: ${(data.f1_scores[lastIdx] * 100).toFixed(2)}%</p>
                                    <p><strong>Best Accuracy:</strong> ${(Math.max(...data.accuracies) * 100).toFixed(2)}% (Epoch ${data.epochs[bestAccIdx]})</p>
                                    <p><strong>Best UAR:</strong> ${(Math.max(...data.uars) * 100).toFixed(2)}% (Epoch ${data.epochs[bestUarIdx]})</p>
                                </div>
                            `;
                            
                            // Set the image source
                            modalImage.src = `history/speaker_${speakerId}/training_progress.png?t=${new Date().getTime()}`;
                            
                            // Show the modal
                            modal.style.display = 'block';
                        })
                        .catch(error => {
                            console.error('Error loading speaker metrics:', error);
                            modalTitle.textContent = `Error Loading History for Speaker ${speakerId}`;
                            modalContent.innerHTML = `<p>Could not load metrics for this speaker. Error: ${error.message}</p>`;
                            modalImage.style.display = 'none';
                            modal.style.display = 'block';
                        });
                }
                
                // Function to close the modal
                function closeModal() {
                    document.getElementById('history-modal').style.display = 'none';
                }
                
                // Function to fetch and update current metrics
                function updateDashboard() {
                    const timestamp = new Date().getTime();
                    
                    // Fetch training metrics
                    fetch('current_metrics.json?t=' + timestamp)
                        .then(response => response.json())
                        .then(data => {
                            // Update speaker info
                            document.getElementById('current-speaker').textContent = data.speaker || 'None';
                            document.getElementById('speaker-id').textContent = `Speaker ID: ${data.speaker_id || 'None'}`;
                            
                            if (data.epochs && data.epochs.length > 0) {
                                const lastIndex = data.epochs.length - 1;
                                
                                // Update metrics
                                document.getElementById('train-loss').textContent = 
                                    data.train_losses[lastIndex] !== null ? data.train_losses[lastIndex].toFixed(4) : 'N/A';
                                document.getElementById('val-loss').textContent = 
                                    data.val_losses[lastIndex] !== null ? data.val_losses[lastIndex].toFixed(4) : 'N/A';
                                document.getElementById('accuracy').textContent = 
                                    data.accuracies[lastIndex] !== null ? (data.accuracies[lastIndex] * 100).toFixed(2) + '%' : 'N/A';
                                document.getElementById('uar').textContent = 
                                    data.uars[lastIndex] !== null ? (data.uars[lastIndex] * 100).toFixed(2) + '%' : 'N/A';
                                document.getElementById('f1').textContent = 
                                    data.f1_scores[lastIndex] !== null ? (data.f1_scores[lastIndex] * 100).toFixed(2) + '%' : 'N/A';
                                
                                // Update best metrics
                                if (data.accuracies.some(val => val !== null)) {
                                    const bestAccuracy = Math.max(...data.accuracies.filter(val => val !== null));
                                    const bestAccuracyIndex = data.accuracies.indexOf(bestAccuracy);
                                    document.getElementById('best-accuracy').textContent = (bestAccuracy * 100).toFixed(2) + '%';
                                    document.getElementById('best-accuracy-epoch').textContent = `Epoch: ${data.epochs[bestAccuracyIndex]}`;
                                }
                                
                                if (data.uars.some(val => val !== null)) {
                                    const bestUAR = Math.max(...data.uars.filter(val => val !== null));
                                    const bestUARIndex = data.uars.indexOf(bestUAR);
                                    document.getElementById('best-uar').textContent = (bestUAR * 100).toFixed(2) + '%';
                                    document.getElementById('best-uar-epoch').textContent = `Epoch: ${data.epochs[bestUARIndex]}`;
                                }
                                
                                if (data.f1_scores.some(val => val !== null)) {
                                    const bestF1 = Math.max(...data.f1_scores.filter(val => val !== null));
                                    const bestF1Index = data.f1_scores.indexOf(bestF1);
                                    document.getElementById('best-f1').textContent = (bestF1 * 100).toFixed(2) + '%';
                                    document.getElementById('best-f1-epoch').textContent = `Epoch: ${data.epochs[bestF1Index]}`;
                                }
                                
                                // Update last updated time
                                if (data.last_update) {
                                    document.getElementById('updated-at').textContent = 'Last updated: ' + data.last_update;
                                }
                                
                                // Update plots with cache-busting
                                const plotsImg = document.getElementById('plots-img');
                                plotsImg.src = 'plots/training_progress.png?t=' + timestamp;
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching training metrics:', error);
                        });
                    
                    // Fetch batch metrics
                    fetch('batch_metrics.json?t=' + timestamp)
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('epoch-info').textContent = 
                                `Epoch: ${data.current_epoch} / ?`;
                            
                            document.getElementById('batch-info').textContent = 
                                `Batch: ${data.current_batch} / ${data.total_batches}`;
                            
                            // Update progress bar
                            const percentage = (data.current_batch / data.total_batches) * 100;
                            const progressFill = document.getElementById('progress-fill');
                            progressFill.style.width = percentage + '%';
                            progressFill.textContent = percentage.toFixed(1) + '%';
                        })
                        .catch(error => {
                            console.error('Error fetching batch metrics:', error);
                        });
                }
                
                // Update dashboard initially
                updateDashboard(true); // Force first update

                
                // Set interval to update dashboard every 5 seconds
                setInterval(updateDashboard, 5000);
                
                // Close the modal when clicking outside of it
                window.onclick = function(event) {
                    const modal = document.getElementById('history-modal');
                    if (event.target == modal) {
                        modal.style.display = 'none';
                    }
                }
            </script>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(self.dashboard_dir / 'dashboard.html', 'w') as f:
            f.write(html_content)
    def finish(self):
        """Finish the current speaker run and clean up"""
        if self.current_speaker_id is not None:
            self._archive_current_run()
            self.current_speaker_id = None
            self.current_speaker = None
            print("Archived current run and reset dashboard")
            
    def export_results(self, export_dir=None):
        """
        Export all dashboard results to a specified directory.
        
        Args:
            export_dir (str, optional): Directory to save results. If None, creates a timestamped
                                       directory in the base_dir/exports folder.
        
        Returns:
            str: Path to the export directory
        """
        # Create export directory if not specified
        if export_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exports_dir = self.base_dir / 'exports'
            exports_dir.mkdir(exist_ok=True)
            export_dir = exports_dir / f"dashboard_export_{timestamp}"
        else:
            export_dir = Path(export_dir)
        
        export_dir.mkdir(exist_ok=True, parents=True)
        
        # Export current summary data
        self._export_summary(export_dir)
        
        # Export history for each speaker
        for speaker_data in self.speaker_history:
            speaker_id = speaker_data['speaker_id']
            self._export_speaker_data(speaker_id, export_dir)
        
        # Also export the current speaker if it exists
        if self.current_speaker_id is not None:
            self._archive_current_run()  # Make sure it's archived first
            self._export_speaker_data(self.current_speaker_id, export_dir)
        
        print(f"Dashboard results exported to {export_dir}")
        return str(export_dir)

    def _export_summary(self, export_dir):
        """Create a summary report of all speakers"""
        # Create summary dataframe
        data = []
        for speaker in self.speaker_history:
            data.append({
                'Speaker ID': speaker['speaker_id'],
                'Speaker Name': speaker['speaker_name'],
                'Training Date': speaker['timestamp'],
                'Best Accuracy': speaker['metrics']['best_accuracy'],
                'Best UAR': speaker['metrics']['best_uar'],
                'Best F1': speaker['metrics']['best_f1'],
                'Epochs Trained': speaker['metrics']['epochs_trained']
            })
        
        if data:
            # Create summary dataframe
            df = pd.DataFrame(data)
            
            # Save as CSV
            df.to_csv(export_dir / 'speaker_summary.csv', index=False)
            
            # Create a summary plot
            plt.figure(figsize=(12, 8))
            
            # Set plot data
            speaker_ids = [str(row['Speaker ID']) for row in data]
            accuracies = [row['Best Accuracy'] * 100 if row['Best Accuracy'] is not None else 0 for row in data]
            uars = [row['Best UAR'] * 100 if row['Best UAR'] is not None else 0 for row in data]
            f1s = [row['Best F1'] * 100 if row['Best F1'] is not None else 0 for row in data]
            
            # Plot bar chart
            bar_width = 0.25
            x = range(len(speaker_ids))
            
            plt.bar([i - bar_width for i in x], accuracies, width=bar_width, label='Accuracy', color='green')
            plt.bar(x, uars, width=bar_width, label='UAR', color='purple')
            plt.bar([i + bar_width for i in x], f1s, width=bar_width, label='F1', color='cyan')
            
            plt.xlabel('Speaker ID')
            plt.ylabel('Score (%)')
            plt.title('Performance Summary Across All Speakers')
            plt.xticks(x, speaker_ids)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(export_dir / 'performance_summary.png')
            plt.close()

    def _export_speaker_data(self, speaker_id, export_dir):
        """Export complete data for a single speaker"""
        speaker_dir = self.history_dir / f"speaker_{speaker_id}"
        if not speaker_dir.exists():
            return
        
        # Create speaker directory in export location
        export_speaker_dir = export_dir / f"speaker_{speaker_id}"
        export_speaker_dir.mkdir(exist_ok=True)
        
        # Copy metrics JSON
        if (speaker_dir / "metrics.json").exists():
            shutil.copy(speaker_dir / "metrics.json", export_speaker_dir / "metrics.json")
            
            # Also create a CSV from the JSON for easier analysis
            try:
                with open(speaker_dir / "metrics.json", 'r') as f:
                    metrics = json.load(f)
                
                # Create dataframe from metrics
                df = pd.DataFrame({
                    'epoch': metrics['epochs'],
                    'train_loss': metrics['train_losses'],
                    'val_loss': metrics['val_losses'],
                    'accuracy': metrics['accuracies'],
                    'uar': metrics['uars'],
                    'f1_score': metrics['f1_scores']
                })
                
                df.to_csv(export_speaker_dir / "training_metrics.csv", index=False)
            except Exception as e:
                print(f"Error converting metrics to CSV for speaker {speaker_id}: {e}")
        
        # Copy training progress plot
        if (speaker_dir / "training_progress.png").exists():
            shutil.copy(speaker_dir / "training_progress.png", export_speaker_dir / "training_progress.png")