
import torch
import torch.nn as nn
import os
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt

from model import GPT, GPTConfig


def load_dataset_info(data_dir):
    """Load vocabulary size from meta.pkl."""
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"  Found vocab_size = {vocab_size} in {meta_path}")
    else:
        vocab_size = 50304  # GPT-2 default
        print(f"  No meta.pkl found, defaulting vocab_size = {vocab_size}")
    return vocab_size


def get_batch(data_dir, split, batch_size, block_size, device):
    """
    Load a batch from memory-mapped binary file.
    Same approach as train.py - recreate memmap each call to avoid memory leak.
    """
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if 'cuda' in device:
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


class PAdaptationMetaLearner:
    """Meta-learner that inherits plasticity masks P, not weights."""

    def __init__(self, model_config, base_dir='p_adaptation_experiments'):
        self.model_config = model_config
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.P = None  # Plasticity mask (inherited across tasks)
        self.P_history = []  # Track P evolution
        self.loss_history = []  # Track convergence

    def initialize_P(self, model):
        """Init plasticity mask P with uniform values for transformer params."""
        P = {}
        for name, param in model.named_parameters():
            # Skip embeddings, meta-learn transformer only
            if 'transformer.h.' in name or 'transformer.ln_f' in name:
                P[name] = torch.ones_like(param.data)

        print(f"Initialized P with {len(P)} parameter groups")
        return P

    def reset_model_weights(self, model):
        """Reset transformer weights. Keeps embeddings (stripped later for new vocab)."""
        print("Resetting model weights to random...")

        for name, param in model.named_parameters():
            if 'transformer.h.' in name or 'transformer.ln_f' in name:
                if 'weight' in name:
                    if len(param.shape) >= 2:
                        # Linear layers: normal init with proper scaling
                        std = 0.02
                        if name.endswith('c_proj.weight'):
                            std *= (2 * self.model_config.n_layer) ** -0.5
                        nn.init.normal_(param.data, mean=0.0, std=std)
                    else:
                        # 1D weights (e.g., LayerNorm): ones
                        nn.init.ones_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    def train_task_with_P(self, model, data_dir, P, train_config):
        """Train with P-masked gradients using get_batch. Returns final params, trajectory, losses."""
        device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.train()

        # Training config
        batch_size = train_config.get('batch_size', 32)
        block_size = self.model_config.block_size
        learning_rate = train_config.get('learning_rate', 1e-2) # SGD needs higher LR
        max_iters = train_config.get('max_iters', 1000)
        checkpoint_interval = train_config.get('checkpoint_interval', 100)

        # Optimizer
        # CHANGED: Switched to SGD + Momentum. Adam normalizes out P scaling.
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=train_config.get('weight_decay', 0.0)
        )

        # Store initial params
        initial_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if name in P
        }

        # Trajectory checkpoints
        trajectory = []
        losses = []

        print(f"Training with P-masked gradients for {max_iters} iterations...")

        for iter_num in range(max_iters):
            # Get batch using memmap approach
            x, y = get_batch(data_dir, 'train', batch_size, block_size, device)

            # Forward pass
            logits, loss = model(x, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Apply P mask to gradients (controls learning per parameter)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in P and param.grad is not None:
                        param.grad *= P[name]

            # Gradient clipping
            if train_config.get('grad_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])

            optimizer.step()

            # Track loss
            losses.append((iter_num, loss.item()))

            # Checkpoint for P gradient computation
            if iter_num % checkpoint_interval == 0 or iter_num == max_iters - 1:
                checkpoint = {
                    'iter': iter_num,
                    'params': {
                        name: param.data.clone()
                        for name, param in model.named_parameters()
                        if name in P
                    },
                    'loss': loss.item()
                }
                trajectory.append(checkpoint)

            # Logging
            if iter_num % 100 == 0:
                print(f"  Iter {iter_num}/{max_iters} | Loss: {loss.item():.4f}")

        # Final params
        final_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if name in P
        }

        return final_params, trajectory, losses, initial_params

    def compute_P_gradient(self, initial_params, trajectory, model, data_dir, P, train_config):
        """Compute P gradient from trajectory: ΔP ∝ Σ P^{-1} ⊙ (Θ_n - Θ_0) ⊙ ∇L|_n"""
        device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = train_config.get('batch_size', 32)
        block_size = self.model_config.block_size
        model.eval()

        P_grad = {name: torch.zeros_like(P[name]) for name in P.keys()}

        print(f"Computing P gradient from {len(trajectory)} checkpoints...")

        # For each checkpoint, compute gradient contribution
        for checkpoint in trajectory:
            # Load checkpoint params temporarily
            checkpoint_params = checkpoint['params']
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in checkpoint_params:
                        param.data.copy_(checkpoint_params[name])

            # Compute gradient at this checkpoint
            # Average over a few batches for stability
            grad_accumulator = {name: torch.zeros_like(P[name]) for name in P.keys()}
            n_grad_batches = train_config.get('n_grad_batches', 5)

            for i in range(n_grad_batches):
                x, y = get_batch(data_dir, 'train', batch_size, block_size, device)

                # Compute gradient
                model.zero_grad()
                logits, loss = model(x, y)
                loss.backward()

                # Accumulate gradients
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in P and param.grad is not None:
                            grad_accumulator[name] += param.grad / n_grad_batches

            # Compute contribution: P^{-1} ⊙ (Θ_n - Θ_0) ⊙ ∇L
            with torch.no_grad():
                for name in P.keys():
                    P_inv = 1.0 / (P[name] + 1e-8)
                    theta_diff = checkpoint_params[name] - initial_params[name]
                    grad_term = grad_accumulator[name]
                    P_grad[name] += P_inv * theta_diff * grad_term

        return P_grad

    def update_and_renormalize_P(self, P, P_grad, meta_lr):
        """Update P and renormalize: P* = (Tr(P) / Tr(P²)) * P"""
        
        # CHANGED: Added gradient clipping for evolutionary stability
        max_norm = 1.0
        total_norm = torch.norm(torch.stack([torch.norm(g) for g in P_grad.values()]))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            print(f"  Clipping P_grad norm: {total_norm:.2f} -> {max_norm}")
            for g in P_grad.values():
                g.mul_(scale)

        P_new = {}
        for name in P.keys():
            # CHANGED: Changed '+' to '-' because we minimize loss (Fitness = -Loss)
            P_new[name] = P[name] - meta_lr * P_grad[name]
            P_new[name] = torch.clamp(P_new[name], min=1e-6)  # Keep positive

        # Renormalize
        trace_P = sum(p.sum().item() for p in P_new.values())
        trace_P2 = sum((p**2).sum().item() for p in P_new.values())
        scale = trace_P / (trace_P2 + 1e-8)

        print(f"  Renormalizing P: scale factor = {scale:.6f}")

        for name in P_new.keys():
            P_new[name] = scale * P_new[name]

        return P_new

    def save_P(self, P, iteration, save_dir):
        """Save plasticity mask"""
        save_path = os.path.join(save_dir, f'P_iter_{iteration}.pt')
        torch.save(P, save_path)
        print(f"  Saved P to {save_path}")

    def run_P_metalearning(self,
                           dataset_dirs,
                           train_config,
                           num_iterations=10,
                           meta_learning_rate=0.1):
        """
        Meta-learn P across language tasks. Each task: reset weights, train with P,
        compute P gradient, update P. Only P is inherited.

        Args:
            dataset_dirs: List of data directories, each containing train.bin and meta.pkl
            train_config: Training hyperparameters
            num_iterations: Number of meta-iterations
            meta_learning_rate: P update step size
        """
        print("\n" + "="*80)
        print("P-ONLY META-LEARNING")
        print(f"Meta-learning plasticity across {num_iterations} tasks")
        print(f"Meta-learning rate: {meta_learning_rate}")
        print("="*80 + "\n")

        # Initialize model (will be reset each task)
        model = None

        for iteration in range(num_iterations):
            dataset_idx = iteration % len(dataset_dirs)
            data_dir = dataset_dirs[dataset_idx]

            print(f"\n{'='*80}")
            print(f"META-ITERATION {iteration + 1}/{num_iterations}")
            print(f"Dataset: {data_dir}")

            # Load vocab size for this dataset
            vocab_size = load_dataset_info(data_dir)
            print(f"Vocab size: {vocab_size}")
            print(f"{'='*80}\n")

            # Create/reset model for this task
            if model is None:
                # First iteration: create model and initialize P
                self.model_config.vocab_size = vocab_size
                model = GPT(self.model_config)
                self.P = self.initialize_P(model)
                print("Created initial model and initialized P")
            else:
                # Reset weights each task
                self.reset_model_weights(model)

                # Update vocab size if needed
                if model.config.vocab_size != vocab_size:
                    model.update_vocab_size(vocab_size)
                else:
                    model.strip_embeddings()

            # Train with current P
            print("\nPhase 1: Training with P-masked gradients...")
            final_params, trajectory, losses, initial_params = self.train_task_with_P(
                model, data_dir, self.P, train_config
            )

            # Track convergence
            final_loss = np.mean([loss for _, loss in losses[-100:]])
            self.loss_history.append({
                'iteration': iteration,
                'dataset': data_dir,
                'final_loss': final_loss
            })

            # Compute P gradient
            print("\nPhase 2: Computing P gradient from trajectory...")
            P_grad = self.compute_P_gradient(
                initial_params, trajectory, model, data_dir, self.P, train_config
            )

            # Update P
            print("\nPhase 3: Updating and renormalizing P...")
            self.P = self.update_and_renormalize_P(self.P, P_grad, meta_learning_rate)

            # Save P
            iter_dir = os.path.join(self.base_dir, f'iteration_{iteration + 1}')
            os.makedirs(iter_dir, exist_ok=True)
            self.save_P(self.P, iteration + 1, iter_dir)

            # Track P statistics
            P_stats = self.compute_P_statistics(self.P)
            self.P_history.append({
                'iteration': iteration,
                'mean': P_stats['mean'],
                'std': P_stats['std'],
                'min': P_stats['min'],
                'max': P_stats['max']
            })

            print(f"\nP Statistics: mean={P_stats['mean']:.4f}, std={P_stats['std']:.4f}, "
                  f"min={P_stats['min']:.4f}, max={P_stats['max']:.4f}")

            # Save progress
            self.save_history()
            self.plot_progress()

        print("\n" + "="*80)
        print("P-ONLY META-LEARNING COMPLETE")
        print("="*80 + "\n")

        return self.P

    def compute_P_statistics(self, P):
        """Compute statistics of P across all parameters"""
        all_P_values = torch.cat([p.flatten() for p in P.values()])
        return {
            'mean': all_P_values.mean().item(),
            'std': all_P_values.std().item(),
            'min': all_P_values.min().item(),
            'max': all_P_values.max().item()
        }

    def save_history(self):
        """Save training history"""
        with open(os.path.join(self.base_dir, 'P_history.json'), 'w') as f:
            json.dump(self.P_history, f, indent=2)

        with open(os.path.join(self.base_dir, 'loss_history.json'), 'w') as f:
            json.dump(self.loss_history, f, indent=2)

    def plot_progress(self):
        """Plot P evolution and convergence"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: P statistics over time
        if self.P_history:
            iters = [h['iteration'] for h in self.P_history]
            means = [h['mean'] for h in self.P_history]
            stds = [h['std'] for h in self.P_history]
            mins = [h['min'] for h in self.P_history]
            maxs = [h['max'] for h in self.P_history]

            ax1.plot(iters, means, 'b-', linewidth=2, label='Mean')
            ax1.fill_between(iters,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.3, color='b')
            ax1.plot(iters, mins, 'r--', alpha=0.5, label='Min')
            ax1.plot(iters, maxs, 'g--', alpha=0.5, label='Max')
            ax1.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Uniform')
            ax1.set_xlabel('Meta-Iteration')
            ax1.set_ylabel('P Values')
            ax1.set_title('Plasticity Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Loss convergence
        if self.loss_history:
            iters = [h['iteration'] for h in self.loss_history]
            losses = [h['final_loss'] for h in self.loss_history]

            ax2.plot(iters, losses, 'o-', linewidth=2, markersize=8)
            ax2.set_xlabel('Meta-Iteration')
            ax2.set_ylabel('Final Training Loss')
            ax2.set_title('Language Learning Convergence')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'p_adaptation_progress.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Demo P-adaptation: inherit plasticity, not structure."""
    print("P-Adaptation Meta-Learning Example")
    print("="*80)

    # Configuration
    model_config = GPTConfig(
        vocab_size=50304,  # Will be updated per dataset
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.0,
        bias=True
    )

    # CHANGED: SGD hyperparameters
    train_config = {
        'batch_size': 32,
        'learning_rate': 1e-2,       # Higher LR for SGD
        'max_iters': 1000,
        'checkpoint_interval': 100,  # Checkpoint every 100 iters
        'n_grad_batches': 10,        # Increased for stable selection pressure
        'weight_decay': 0.0,         # SGD doesn't strictly need L2 here
        'grad_clip': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Each dataset directory should contain train.bin, val.bin, and meta.pkl
    # These should be created using the prepare.py script for each language
    dataset_dirs = [
        'data/english',
        'data/japanese',
        'data/arabic',
        'data/finnish'
    ]

    # Check if data exists
    print("\nChecking for dataset directories...")
    missing = []
    for data_dir in dataset_dirs:
        train_bin = os.path.join(data_dir, 'train.bin')
        meta_pkl = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(train_bin):
            missing.append(train_bin)
        if not os.path.exists(meta_pkl):
            missing.append(meta_pkl)

    if missing:
        print("\nERROR: Required dataset files not found!")
        print("Missing files:")
        for f in missing:
            print(f"  - {f}")
        print("\nFor each language, you need to:")
        print("1. Create a directory (e.g., data/english/)")
        print("2. Run the prepare.py script to generate train.bin and meta.pkl")
        print("3. See data/openwebtext/prepare.py for an example")
        return

    print("All dataset directories found!")

    # Create meta-learner
    meta_learner = PAdaptationMetaLearner(model_config, base_dir='p_adaptation_results')

    # Run P-only meta-learning
    final_P = meta_learner.run_P_metalearning(
        dataset_dirs=dataset_dirs,
        train_config=train_config,
        num_iterations=8,  # 2 passes through 4 languages
        meta_learning_rate=0.1
    )

    print("\nFinal plasticity mask learned!")
    print(f"Results saved to: {meta_learner.base_dir}/")
    print("\nNext steps:")
    print("1. Test final_P on unseen language")
    print("2. Compare to random P (no adaptation)")
    print("3. Compare to θ₀-only and both adaptations")


if __name__ == "__main__":
    main()