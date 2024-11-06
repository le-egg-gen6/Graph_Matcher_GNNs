from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_training_metrics(metrics, figsize=(15, 10), save_path=None):
    """
    Plot training metrics including loss curves and learning rate.
    
    Args:
        metrics (dict): Training metrics dictionary containing:
            - losses: List of batch losses
            - epoch_losses: List of epoch average losses
            - lr_history: List of learning rates
            - metrics: Dict containing step, loss, learning_rate arrays
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the plot
    """
    plt.style.use('seaborn')
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Training Loss (by steps)
    ax1 = fig.add_subplot(gs[0, 0])
    steps = metrics['metrics']['step']
    losses = metrics['metrics']['loss']
    
    # Plot raw losses with light color
    ax1.plot(steps, losses, 'lightblue', alpha=0.3, label='Batch Loss')
    
    # Plot smoothed losses
    window_size = min(50, len(losses)//10)
    if window_size > 0:
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = steps[window_size-1:]
        ax1.plot(smoothed_steps, smoothed_losses, 'blue', label='Smoothed Loss')
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss by Steps')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Learning Rate
    ax2 = fig.add_subplot(gs[0, 1])
    lr_history = metrics['lr_history']
    ax2.plot(steps, lr_history, 'green', label='Learning Rate')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # 3. Epoch Average Loss
    ax3 = fig.add_subplot(gs[1, 0])
    epochs = range(1, len(metrics['epoch_losses']) + 1)
    ax3.plot(epochs, metrics['epoch_losses'], 'red', marker='o', label='Epoch Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Average Loss')
    ax3.set_title('Average Loss per Epoch')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # 4. Loss Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(losses, bins=50, color='purple', alpha=0.7)
    ax4.axvline(np.mean(losses), color='red', linestyle='dashed', label='Mean Loss')
    ax4.axvline(np.median(losses), color='green', linestyle='dashed', label='Median Loss')
    ax4.set_xlabel('Loss Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Loss Distribution')
    ax4.legend()
    
    # Add overall title
    fig.suptitle('Training Metrics Overview', fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    
    return fig

def plot_comparative_metrics(metrics_list, labels, figsize=(15, 5), save_path=None):
    """
    Plot comparative metrics from multiple training runs.
    
    Args:
        metrics_list (list): List of metrics dictionaries from different runs
        labels (list): Labels for each run
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the plot
    """
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(metrics_list)))
    
    # 1. Training Loss Comparison
    for metrics, label, color in zip(metrics_list, labels, colors):
        steps = metrics['metrics']['step']
        losses = metrics['metrics']['loss']
        
        # Plot smoothed losses
        window_size = min(50, len(losses)//10)
        if window_size > 0:
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = steps[window_size-1:]
            ax1.plot(smoothed_steps, smoothed_losses, color=color, label=label)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Smoothed Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Learning Rate Comparison
    for metrics, label, color in zip(metrics_list, labels, colors):
        steps = metrics['metrics']['step']
        lr_history = metrics['lr_history']
        ax2.plot(steps, lr_history, color=color, label=label)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule Comparison')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Comparative plot saved to {save_path}")
    
    return fig

# # Create sample metrics data
# def create_sample_metrics(n_steps=1000, noise_level=0.2):
#     steps = np.arange(n_steps)
    
#     # Generate decreasing loss with noise
#     base_loss = 2 * np.exp(-steps / 300) + 0.5
#     noise = np.random.normal(0, noise_level, n_steps)
#     losses = base_loss + noise
#     losses = np.abs(losses)  # Ensure losses are positive
    
#     # Generate learning rate with step decay
#     lr_base = 0.01
#     lr_history = lr_base * np.power(0.1, steps // 300)
    
#     # Calculate epoch losses (assuming 100 steps per epoch)
#     n_epochs = n_steps // 100
#     epoch_losses = [np.mean(losses[i*100:(i+1)*100]) for i in range(n_epochs)]
    
#     return {
#         'metrics': {
#             'step': steps,
#             'loss': losses,
#             'learning_rate': lr_history
#         },
#         'lr_history': lr_history,
#         'epoch_losses': epoch_losses,
#         'losses': losses.tolist()
#     }

# # Create sample data for single run
# metrics_single = create_sample_metrics(n_steps=1000, noise_level=0.2)

# # Create sample data for multiple runs with different characteristics
# metrics_list = [
#     create_sample_metrics(n_steps=1000, noise_level=0.2),  # Normal training
#     create_sample_metrics(n_steps=1000, noise_level=0.4),  # More noisy training
# ]
# labels = ['Normal Training', 'Noisy Training']

# # Plot single training run
# fig1 = plot_training_metrics(metrics_single, figsize=(15, 10))
# plt.show()

# # Plot comparative metrics
# fig2 = plot_comparative_metrics(metrics_list, labels, figsize=(15, 5))
# plt.show()

# create_sample_metrics()
