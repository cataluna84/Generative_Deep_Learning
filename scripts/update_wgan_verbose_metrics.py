#!/usr/bin/env python3
"""
Update WGAN.py with enhanced verbose metrics logging.

This script modifies the WGAN class to:
1. Import the new metrics modules
2. Add metrics history dictionary
3. Update train() method with verbose output
4. Add quality metrics computation every 100 epochs

Usage:
    python scripts/update_wgan_verbose_metrics.py
"""

import json
import os
import re


def update_wgan_model():
    """Update WGAN.py with verbose metrics."""
    wgan_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "v1",
        "src",
        "models",
        "WGAN.py"
    )
    wgan_path = os.path.normpath(wgan_path)

    with open(wgan_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add imports at the top (after existing imports)
    import_addition = '''
# =============================================================================
# METRICS IMPORTS
# =============================================================================
# Import GAN metrics utilities for verbose training output
try:
    from utils.gan.metrics import (
        collect_epoch_metrics,
        format_verbose_output,
        EpochTimer,
    )
    from utils.gan.quality_metrics import (
        collect_quality_metrics,
        format_quality_output,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

'''

    # Find position after matplotlib import
    import_marker = "import matplotlib.pyplot as plt"
    if import_marker in content:
        content = content.replace(
            import_marker,
            import_marker + "\n" + import_addition
        )
        print("✓ Added metrics imports")

    # 2. Add metrics_history attribute in __init__
    init_addition = '''
        # Initialize metrics history for verbose logging
        self.metrics_history = {
            'wasserstein_dist': [],
            'dg_ratio': [],
            'clip_ratio': [],
            'epoch_time': [],
            'critic_weight_mean': [],
            'critic_weight_std': [],
            'generator_weight_mean': [],
            'generator_weight_std': [],
        }

'''

    # Find position after self.epoch = 0
    epoch_marker = "self.epoch = 0"
    if epoch_marker in content and "self.metrics_history" not in content:
        content = content.replace(
            epoch_marker,
            epoch_marker + "\n" + init_addition
        )
        print("✓ Added metrics_history attribute")

    # 3. Replace the train() method with verbose version
    new_train_method = '''    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 10
        , n_critic = 5
        , clip_threshold = 0.01
        , using_generator = False
        , verbose = True
        , quality_metrics_every = 100):
        """
        Train the WGAN model with enhanced metrics logging.
        
        Args:
            x_train: Training data (numpy array or generator).
            batch_size: Number of samples per batch.
            epochs: Number of epochs to train.
            run_folder: Path to save outputs.
            print_every_n_batches: Save images every N epochs.
            n_critic: Number of critic updates per generator update.
            clip_threshold: Weight clipping threshold for WGAN.
            using_generator: If True, x_train is a data generator.
            verbose: If True, use verbose output with all metrics.
            quality_metrics_every: Compute FID/IS every N epochs (0 to disable).
        """
        import time
        
        for epoch in range(self.epoch, self.epoch + epochs):
            epoch_start = time.time()

            for _ in range(n_critic):
                d_loss = self.train_critic(x_train, batch_size, clip_threshold, using_generator)

            g_loss = self.train_generator(batch_size)
            
            epoch_time = time.time() - epoch_start
            
            # Store losses
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)
            
            # Verbose output with all metrics
            if verbose and METRICS_AVAILABLE:
                # Collect comprehensive metrics
                metrics = collect_epoch_metrics(
                    critic=self.critic,
                    generator=self.generator,
                    d_loss=d_loss,
                    g_loss=g_loss,
                    clip_threshold=clip_threshold,
                    epoch_time=epoch_time,
                    loss_history=self.d_losses if len(self.d_losses) > 1 else None
                )
                
                # Store metrics history
                for key in ['wasserstein_dist', 'dg_ratio', 'clip_ratio', 'epoch_time']:
                    if key == 'wasserstein_dist':
                        self.metrics_history[key].append(metrics.get('wasserstein_distance', 0))
                    else:
                        self.metrics_history[key].append(metrics.get(key, 0))
                
                for key in ['critic_weight_mean', 'critic_weight_std', 
                           'generator_weight_mean', 'generator_weight_std']:
                    self.metrics_history[key].append(metrics.get(key, 0))
                
                # Print verbose output
                output = format_verbose_output(epoch, self.epoch + epochs, metrics)
                print(output)
                
                # Quality metrics every N epochs
                if quality_metrics_every > 0 and epoch % quality_metrics_every == 0 and epoch > 0:
                    print("\\n  Computing quality metrics...")
                    try:
                        # Generate sample images
                        noise = np.random.normal(0, 1, (100, self.z_dim))
                        gen_imgs = self.generator.predict(noise, verbose=0)
                        
                        # Get sample real images
                        if using_generator:
                            real_imgs = next(x_train)[0]
                        else:
                            idx = np.random.randint(0, x_train.shape[0], 100)
                            real_imgs = x_train[idx]
                        
                        quality = collect_quality_metrics(
                            real_images=real_imgs,
                            fake_images=gen_imgs,
                            compute_fid=True,
                            compute_is=True
                        )
                        print(format_quality_output(quality))
                    except Exception as e:
                        print(f"  ⚠ Quality metrics failed: {e}")
            else:
                # Simple output (original behavior)
                print ("%d [D loss: (%.3f)(R %.3f, F %.3f)]  [G loss: %.3f] " % (
                    epoch, d_loss[0], d_loss[1], d_loss[2], g_loss))

            # If at save interval => save generated image samples
            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.weights.h5' % (epoch)))
                self.model.save_weights(os.path.join(run_folder, 'weights/weights.weights.h5'))
                self.save_model(run_folder)
            
            self.epoch+=1

'''

    # Find and replace the train method
    train_pattern = r'    def train\(self, x_train, batch_size, epochs, run_folder.*?self\.epoch\+=1'
    
    if re.search(train_pattern, content, re.DOTALL):
        content = re.sub(train_pattern, new_train_method.rstrip(), content, flags=re.DOTALL)
        print("✓ Updated train() method with verbose metrics")
    else:
        print("⚠ Could not find train() method pattern")

    # Save the updated file
    with open(wgan_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Saved updated WGAN.py: {wgan_path}")


if __name__ == "__main__":
    update_wgan_model()
