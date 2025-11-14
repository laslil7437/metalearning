import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from model import GPT, GPTConfig


class TextDataset(Dataset):
    """Text dataset for language training"""
    def __init__(self, text_file, tokenizer, block_size=256, max_samples=None):
        print(f"Loading dataset from {text_file}...")

        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        self.tokens = tokenizer.encode(text)

        if max_samples:
            max_tokens = max_samples * block_size
            self.tokens = self.tokens[:max_tokens]

        print(f"Dataset loaded: {len(self.tokens)} tokens")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class SimpleTokenizer:
    """Minimal character-level tokenizer for testing"""
    def __init__(self, vocab=None):
        if vocab is None:
            # Default ASCII vocab
            self.char_to_idx = {chr(i): i for i in range(256)}
            self.idx_to_char = {i: chr(i) for i in range(256)}
        else:
            self.char_to_idx = vocab
            self.idx_to_char = {v: k for k, v in vocab.items()}

    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(t, '?') for t in tokens])

    def get_vocab_size(self):
        return len(self.char_to_idx)


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
            if 'blocks' in name or 'ln_f' in name:
                P[name] = torch.ones_like(param.data)

        print(f"Initialized P with {len(P)} parameter groups")
        return P

    def reset_model_weights(self, model):
        """Reset transformer weights. Keeps embeddings (stripped later for new vocab)."""
        print("Resetting model weights to random...")

        for name, param in model.named_parameters():
            if 'blocks' in name or 'ln_f' in name:
                if 'weight' in name:
                    if len(param.shape) >= 2:
                        # Linear layers: normal init with proper scaling
                        std = 0.02
                        if hasattr(param, 'NANOGPT_SCALE_INIT'):
                            std *= (2 * self.model_config.n_layer) ** -0.5
                        nn.init.normal_(param.data, mean=0.0, std=std)
                    else:
                        # 1D weights (e.g., LayerNorm): ones
                        nn.init.ones_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    def train_task_with_P(self, model, dataset, P, train_config):
        """Train with P-masked gradients. Returns final params, trajectory, losses."""
        device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.train()

        # Training config
        batch_size = train_config.get('batch_size', 32)
        learning_rate = train_config.get('learning_rate', 3e-4)
        max_iters = train_config.get('max_iters', 1000)
        checkpoint_interval = train_config.get('checkpoint_interval', 100)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=train_config.get('weight_decay', 0.1)
        )

        # DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
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
        train_iter = iter(train_loader)

        for iter_num in range(max_iters):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

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

    def compute_P_gradient(self, initial_params, trajectory, model, dataset, P, train_config):
        """Compute P gradient from trajectory: ΔP ∝ Σ P^{-1} ⊙ (Θ_n - Θ_0) ⊙ ∇L|_n"""
        device = train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
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

            dataloader = DataLoader(dataset, batch_size=train_config.get('batch_size', 32), shuffle=True)
            for i, (x, y) in enumerate(dataloader):
                if i >= n_grad_batches:
                    break

                x, y = x.to(device), y.to(device)

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
        P_new = {}
        for name in P.keys():
            P_new[name] = P[name] + meta_lr * P_grad[name]
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
                           dataset_paths,
                           tokenizers,
                           train_config,
                           num_iterations=10,
                           meta_learning_rate=0.1):
        """
        Meta-learn P across language tasks. Each task: reset weights, train with P,
        compute P gradient, update P. Only P is inherited.
        """
        print("\n" + "="*80)
        print("P-ONLY META-LEARNING")
        print(f"Meta-learning plasticity across {num_iterations} tasks")
        print(f"Meta-learning rate: {meta_learning_rate}")
        print("="*80 + "\n")

        # Initialize model (will be reset each task)
        model = None

        for iteration in range(num_iterations):
            dataset_idx = iteration % len(dataset_paths)
            dataset_path = dataset_paths[dataset_idx]
            tokenizer = tokenizers[dataset_idx]

            print(f"\n{'='*80}")
            print(f"META-ITERATION {iteration + 1}/{num_iterations}")
            print(f"Language: {dataset_path}")
            print(f"Vocab size: {tokenizer.get_vocab_size()}")
            print(f"{'='*80}\n")

            # Create/reset model for this task
            if model is None:
                # First iteration: create model and initialize P
                self.model_config.vocab_size = tokenizer.get_vocab_size()
                model = GPT(self.model_config)
                self.P = self.initialize_P(model)
                print("Created initial model and initialized P")
            else:
                # Reset weights each task
                self.reset_model_weights(model)

                # Update vocab size if needed
                if model.config.vocab_size != tokenizer.get_vocab_size():
                    model.update_vocab_size(tokenizer.get_vocab_size())
                else:
                    model.strip_embeddings()

            # Load dataset
            dataset = TextDataset(
                dataset_path,
                tokenizer,
                block_size=self.model_config.block_size,
                max_samples=train_config.get('max_samples', None)
            )

            # Train with current P
            print("\nPhase 1: Training with P-masked gradients...")
            final_params, trajectory, losses, initial_params = self.train_task_with_P(
                model, dataset, self.P, train_config
            )

            # Track convergence
            final_loss = np.mean([loss for _, loss in losses[-100:]])
            self.loss_history.append({
                'iteration': iteration,
                'dataset': dataset_path,
                'final_loss': final_loss
            })

            # Compute P gradient
            print("\nPhase 2: Computing P gradient from trajectory...")
            P_grad = self.compute_P_gradient(
                initial_params, trajectory, model, dataset, self.P, train_config
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
        vocab_size=256,  # Will be updated per language
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256
    )

    train_config = {
        'batch_size': 32,
        'learning_rate': 3e-4,
        'max_iters': 1000,
        'checkpoint_interval': 200,  # Checkpoint every 200 iters
        'n_grad_batches': 5,  # Average over 5 batches for P gradient
        'weight_decay': 0.1,
        'grad_clip': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_samples': 5000  # Limit dataset size for quick testing
    }

    # For demonstration, we'll use simple text files
    # In practice, you'd use real multilingual datasets

    # Example: Create dummy datasets (you should replace with real data)
    print("\nNOTE: This is a demonstration setup.")
    print("Replace dataset_paths and tokenizers with real language data.\n")

    dataset_paths = [
        'data/english.txt',
        'data/japanese.txt',
        'data/arabic.txt',
        'data/finnish.txt'
    ]

    # Create simple tokenizers (replace with proper BPE tokenizers)
    tokenizers = [SimpleTokenizer() for _ in dataset_paths]

    # Check if data exists
    if not all(os.path.exists(p) for p in dataset_paths):
        print("ERROR: Dataset files not found!")
        print("Please create the following files with language data:")
        for p in dataset_paths:
            print(f"  - {p}")
        print("\nOr modify the dataset_paths in the script.")
        return

    # Create meta-learner
    meta_learner = PAdaptationMetaLearner(model_config, base_dir='p_adaptation_results')

    # Run P-only meta-learning
    final_P = meta_learner.run_P_metalearning(
        dataset_paths=dataset_paths,
        tokenizers=tokenizers,
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
