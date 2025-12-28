
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import Callback, ReduceLROnPlateau
import keras.backend as K

class LRFinder(Callback):
    """
    A Keras Callback to find the optimal learning rate.
    
    This callback increases the learning rate exponentially after each batch
    and records the loss. By plotting the loss vs. learning rate, one can 
    identify the point of steepest descent, which is a good candidate for 
    the maximum learning rate.

    Attributes:
        min_lr (float): The minimum learning rate to start the test.
        max_lr (float): The maximum learning rate to end the test.
        steps (int): The total number of steps (batches) for the test.
        beta (float): Smoothing factor for the loss (0 < beta < 1). 
                      Higher beta = more smoothing.
        history (dict): Dictionary to store 'lr' and 'loss' values.
    """
    def __init__(self, min_lr=1e-6, max_lr=1, steps=100, beta=0.98):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = steps
        self.beta = beta
        self.history = {}
        
    def on_train_begin(self, logs=None):
        """Initialize arrays and reset LR at start of training."""
        self.history['lr'] = []
        self.history['loss'] = []
        self.best_loss = 0.
        self.avg_loss = 0.
        self.batch_num = 0
        self.model.optimizer.learning_rate.assign(self.min_lr)
        
    def on_train_batch_end(self, batch, logs=None):
        """Update LR and record loss after each batch."""
        logs = logs or {}
        loss = logs.get('loss')
        
        # Calculate smoothed loss using exponential moving average
        # This helps reduce noise from individual batches
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** (self.batch_num + 1))
        
        # Stop if loss explodes (diverges significantly)
        if self.batch_num > 1 and smoothed_loss > 4 * self.best_loss:
            self.model.stop_training = True
            return

        # Record best loss
        if smoothed_loss < self.best_loss or self.batch_num == 0:
            self.best_loss = smoothed_loss
            
        # Store values
        lr = float(self.model.optimizer.learning_rate)
        self.history['lr'].append(lr)
        self.history['loss'].append(smoothed_loss)
        
        # Calculate and set new Learning Rate
        self.batch_num += 1
        if self.batch_num < self.total_steps:
             # Exponential increase
             new_lr = self.min_lr * (self.max_lr / self.min_lr) ** (self.batch_num / self.total_steps)
             self.model.optimizer.learning_rate.assign(new_lr)

    def get_optimal_lr(self, skip_begin=10, skip_end=5):
        """
        Calculate the optimal learning rate from the recorded history.

        The optimal LR is found at the point of steepest descent in the
        loss vs. learning rate curve. This is calculated by finding the
        learning rate where the gradient (derivative) of the loss is
        most negative.

        Args:
            skip_begin (int): Number of initial steps to skip (often noisy).
            skip_end (int): Number of final steps to skip (often diverged).

        Returns:
            float: The suggested optimal learning rate.
        """
        if len(self.history['lr']) < skip_begin + skip_end + 1:
            raise ValueError(
                "Not enough data points. Run LRFinder for more steps."
            )

        # Get relevant portion of history
        lrs = np.array(self.history['lr'][skip_begin:-skip_end])
        losses = np.array(self.history['loss'][skip_begin:-skip_end])

        # Calculate gradient of loss w.r.t. log(lr)
        # Using log(lr) because LR increases exponentially
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)

        # Find the learning rate at the steepest descent (most negative gradient)
        min_grad_idx = np.argmin(gradients)
        optimal_lr = lrs[min_grad_idx]

        return float(optimal_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plot the loss versus learning rate.
        
        Args:
            n_skip_beginning (int): Number of batches to skip from start.
            n_skip_end (int): Number of batches to skip from end.
        """
        plt.figure(figsize=(10, 5))
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.plot(self.history["lr"][n_skip_beginning:-n_skip_end], 
                 self.history["loss"][n_skip_beginning:-n_skip_end])
        plt.xscale("log")
        
        try:
            best_lr = self.get_optimal_lr(n_skip_beginning, n_skip_end)
            plt.axvline(x=best_lr, color='r', linestyle='--', label=f'Optimal LR: {best_lr:.6f}')
            plt.legend()
            print(f"Optimal Learning Rate: {best_lr}")
        except ValueError:
            print("Could not verify optimal LR from history")
            
        plt.title("Learning Rate Finder: Loss vs LR")
        plt.grid(True, which="both", ls="-")
        plt.show()

def get_lr_scheduler(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-6) -> ReduceLROnPlateau:
    """
    Get a configured ReduceLROnPlateau callback.
    
    This scheduler reduces the learning rate when the validation metric 
    stops improving, allowing the model to take finer steps in the loss landscape.
    
    Args:
        monitor (str): Metric to monitor (e.g., 'val_loss', 'val_accuracy').
        patience (int): Number of epochs with no improvement to wait.
        factor (float): Factor by which the learning rate will be reduced.
        min_lr (float): Lower bound on the learning rate.
        
    Returns:
        ReduceLROnPlateau: Configured Keras callback.
    """
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=1
    )
