
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
    identify the optimal learning rate using various detection methods.

    Usage:
        >>> lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
        >>> model.fit(x_train, y_train, epochs=2, callbacks=[lr_finder])
        >>> lr_finder.plot_loss()
        >>> optimal_lr = lr_finder.get_optimal_lr()  # Uses 'valley' by default
        >>> optimal_lr = lr_finder.get_optimal_lr(method='steepest')  # Override

    Available Selection Methods (for `get_optimal_lr`):
    =====================================================
    
    🔴 'steepest' (Red on plot)
        - AGGRESSIVE: LR at point of steepest descent (minimum gradient).
        - Use when: You want fast training and your model is stable.
        - Risk: May cause training to diverge if too aggressive.
        
    🟠 'recommended' (Orange on plot)
        - BALANCED: steepest_lr / 3 (fastai-style heuristic).
        - Use when: A safer version of steepest; good general choice.
        - Risk: May still be aggressive for unstable models.
        
    🟣 'valley' (Purple on plot) ★ DEFAULT ★
        - ROBUST: LR where loss has dropped 80% toward its minimum.
        - Use when: You want a reliable, data-driven selection.
        - This is the default and generally recommended for most use cases.
        
    🟢 'min_loss_10' (Green on plot)
        - CONSERVATIVE: LR at minimum loss / 10.
        - Use when: Training is unstable or you prefer slow, steady progress.
        - Risk: Training may be slower than necessary.

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

    def _calculate_optimal_lrs(self, skip_begin, skip_end, valley_threshold=0.8):
        """
        Calculate multiple optimal LR candidates using different strategies.
        
        This is an internal helper method. Use `get_optimal_lr()` for the public API.
        
        Args:
            skip_begin (int): Number of initial steps to skip (often noisy).
            skip_end (int): Number of final steps to skip (often diverged).
            valley_threshold (float): Fraction of loss decline for valley detection.
                                      Default 0.8 means find LR at 80% decline.
        
        Returns:
            dict: Dictionary with LR candidates:
                - 'steepest' : 🔴 LR at steepest descent (aggressive)
                - 'recommended': 🟠 steepest / 3 (fastai-style)
                - 'valley'   : 🟣 LR at valley_threshold% decline (robust) ★
                - 'min_loss_10': 🟢 LR at minimum loss / 10 (conservative)
        """
        if len(self.history['lr']) < skip_begin + skip_end + 1:
            raise ValueError("Not enough data points. Run LRFinder for more steps.")

        # Get relevant portion of history
        if skip_end == 0:
            lrs = np.array(self.history['lr'][skip_begin:])
            losses = np.array(self.history['loss'][skip_begin:])
        else:
            lrs = np.array(self.history['lr'][skip_begin:-skip_end])
            losses = np.array(self.history['loss'][skip_begin:-skip_end])

        # Calculate gradient of loss w.r.t. log(lr)
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)

        # Find min loss index and starting loss
        min_loss_idx = np.argmin(losses)
        start_loss = losses[0]
        min_loss = losses[min_loss_idx]
        
        # --- 🔴 Steepest Descent ---
        # Find the LR where the gradient (slope) of the loss curve is most negative.
        # This is typically at the steepest part of the "slide" before the loss valley.
        if min_loss_idx > 0:
            search_gradients = gradients[:min_loss_idx]
            if len(search_gradients) > 0:
                min_grad_idx = np.argmin(search_gradients)
            else:
                min_grad_idx = np.argmin(gradients)
        else:
            min_grad_idx = np.argmin(gradients)
        steepest_lr = float(lrs[min_grad_idx])
        
        # --- 🟠 Recommended (Steepest / 3) ---
        # A common heuristic from fastai: divide the steepest LR by 3 for safety.
        recommended_lr = steepest_lr / 3.0
        
        # --- 🟣 Valley Detection (DEFAULT) ---
        # Find the LR where loss has dropped by valley_threshold% (default 80%)
        # of the total decline from start to minimum. This is more robust to noise.
        total_decline = start_loss - min_loss
        if total_decline > 0:
            target_loss = start_loss - (valley_threshold * total_decline)
            valley_indices = np.where(losses <= target_loss)[0]
            if len(valley_indices) > 0:
                valley_idx = valley_indices[0]
            else:
                valley_idx = min_loss_idx  # Fallback to min loss
        else:
            valley_idx = min_loss_idx  # No decline, fallback
        valley_lr = float(lrs[valley_idx])
        
        # --- 🟢 Min Loss / 10 (Conservative) ---
        # Very safe: take the LR at the minimum loss and divide by 10.
        min_loss_10_lr = float(lrs[min_loss_idx]) / 10.0
        
        return {
            'steepest': steepest_lr,
            'recommended': recommended_lr,
            'valley': valley_lr,
            'min_loss_10': min_loss_10_lr
        }

    def get_optimal_lr(self, skip_begin=10, skip_end=5, method='recommended'):
        """
        Get the optimal learning rate using the specified selection method.
        
        By default, this method uses 'recommended' (steepest / 3), which provides
        a good balance between training speed and stability.

        Args:
            skip_begin (int): Number of initial steps to skip (often noisy).
                              Default: 10.
            skip_end (int): Number of final steps to skip (often diverged).
                            Default: 5.
            method (str): Selection method. Options:
                          - 'steepest'    : 🔴 Aggressive (steepest descent)
                          - 'recommended' : 🟠 Balanced (steepest / 3) ★ DEFAULT ★
                          - 'valley'      : 🟣 Robust (80% decline)
                          - 'min_loss_10' : 🟢 Conservative (min loss / 10)

        Returns:
            float: The selected optimal learning rate.
        
        Examples:
            # Use default (recommended - balanced for most cases)
            >>> lr = lr_finder.get_optimal_lr()
            
            # Use steepest gradient for aggressive training
            >>> lr = lr_finder.get_optimal_lr(method='steepest')
            
            # Use valley for robust, data-driven selection
            >>> lr = lr_finder.get_optimal_lr(method='valley')
            
            # Use conservative approach for unstable models
            >>> lr = lr_finder.get_optimal_lr(method='min_loss_10')
        """
        candidates = self._calculate_optimal_lrs(skip_begin, skip_end)
        
        # Print all candidates with color-coded descriptions
        print("=" * 55)
        print("             LR FINDER RESULTS")
        print("=" * 55)
        print(f"  🔴 Steepest Gradient : {candidates['steepest']:.6f}  (aggressive)")
        print(f"  🟠 Steepest / 3      : {candidates['recommended']:.6f}  (balanced) ★ DEFAULT")
        print(f"  🟣 Valley (80%)      : {candidates['valley']:.6f}  (robust)")
        print(f"  🟢 Min Loss / 10     : {candidates['min_loss_10']:.6f}  (conservative)")
        print("=" * 55)
        print(f"  Selected Method: '{method}' → LR = {candidates.get(method, candidates['recommended']):.6f}")
        print("=" * 55)
        
        selected = candidates.get(method, candidates['recommended'])
        return selected



    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plot the loss versus learning rate with all optimal LR candidates.
        
        Args:
            n_skip_beginning (int): Number of batches to skip from start.
            n_skip_end (int): Number of batches to skip from end.
        """
        plt.figure(figsize=(12, 6))
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (Log Scale)")
        
        # Handle slicing for plot
        if n_skip_end == 0:
            plot_lrs = self.history["lr"][n_skip_beginning:]
            plot_losses = self.history["loss"][n_skip_beginning:]
        else:
            plot_lrs = self.history["lr"][n_skip_beginning:-n_skip_end]
            plot_losses = self.history["loss"][n_skip_beginning:-n_skip_end]
            
        plt.plot(plot_lrs, plot_losses, 'b-', linewidth=2, label='Loss')
        plt.xscale("log")
        
        try:
            candidates = self._calculate_optimal_lrs(n_skip_beginning, n_skip_end)
            
            # Plot all candidates with distinct styles
            # Star (★) indicates the DEFAULT method (recommended = steepest/3)
            plt.axvline(x=candidates['steepest'], color='red', linestyle=':', 
                       alpha=0.6, label=f"Steepest: {candidates['steepest']:.6f}")
            plt.axvline(x=candidates['recommended'], color='orange', linestyle='--', 
                       linewidth=2.5, label=f"Steepest/3: {candidates['recommended']:.6f} ★")
            plt.axvline(x=candidates['valley'], color='purple', linestyle='-', 
                       alpha=0.7, label=f"Valley (80%): {candidates['valley']:.6f}")
            plt.axvline(x=candidates['min_loss_10'], color='green', linestyle='-.', 
                       alpha=0.6, label=f"Min Loss/10: {candidates['min_loss_10']:.6f}")
            
            plt.legend(loc='upper left')
            print(f"Optimal Learning Rate (Recommended): {candidates['recommended']:.6f}")
        except ValueError as e:
            print(f"Could not calculate optimal LR: {e}")
            
        plt.title("Learning Rate Finder: Loss vs LR")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.tight_layout()
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


def get_early_stopping(monitor='loss', patience=10, min_delta=1e-4, 
                       restore_best_weights=True, verbose=1):
    """
    Get a configured EarlyStopping callback.
    
    This callback stops training when a monitored metric has stopped improving,
    preventing overfitting and wasted computation.
    
    Args:
        monitor (str): Metric to monitor (e.g., 'loss', 'val_loss').
                       Use 'loss' for training without validation data.
        patience (int): Number of epochs with no improvement after which 
                        training will be stopped. Default: 10.
        min_delta (float): Minimum change in monitored value to qualify as 
                           an improvement. Default: 1e-4.
        restore_best_weights (bool): Whether to restore model weights from 
                                     the epoch with the best value. Default: True.
        verbose (int): Verbosity mode (0 = silent, 1 = messages). Default: 1.
        
    Returns:
        EarlyStopping: Configured Keras callback.
        
    Example:
        >>> early_stop = get_early_stopping(monitor='loss', patience=10)
        >>> model.fit(x, y, epochs=200, callbacks=[early_stop])
    """
    from keras.callbacks import EarlyStopping
    
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )
