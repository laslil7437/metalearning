"""
Visualization tools for P-adaptation meta-learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


class PVisualizer:
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)

        with open(self.experiment_dir / 'P_history.json', 'r') as f:
            self.P_history = json.load(f)
        with open(self.experiment_dir / 'loss_history.json', 'r') as f:
            self.loss_history = json.load(f)

        self._P_cache = {}

    def _load_P(self, iteration):
        if iteration in self._P_cache:
            return self._P_cache[iteration]

        p_path = self.experiment_dir / f'iteration_{iteration}' / f'P_iter_{iteration}.pt'
        P = torch.load(p_path, map_location='cpu')
        self._P_cache[iteration] = P
        return P

    def _parse_name(self, name):
        """Extract layer number and module type."""
        parts = name.split('.')

        layer = None
        module = None

        if 'transformer.h.' in name:
            layer = int(parts[2])
            if 'attn' in name:
                module = 'attn'
            elif 'mlp' in name:
                module = 'mlp'
            elif 'ln' in name:
                module = 'ln'

        return layer, module

    def plot_evolution(self):
        """P statistics over meta-learning."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        iters = [h['iteration'] for h in self.P_history]
        means = [h['mean'] for h in self.P_history]
        stds = [h['std'] for h in self.P_history]
        mins = [h['min'] for h in self.P_history]
        maxs = [h['max'] for h in self.P_history]

        # Evolution
        ax1.plot(iters, means, label='Mean', linewidth=2)
        ax1.fill_between(iters,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.3, label='±1 std')
        ax1.plot(iters, mins, '--', alpha=0.6, label='Min/Max')
        ax1.plot(iters, maxs, '--', alpha=0.6)
        ax1.axhline(y=1.0, color='red', linestyle=':', label='Uniform')
        ax1.set_xlabel('Meta-iteration')
        ax1.set_ylabel('P value')
        ax1.set_title('Plasticity evolution')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Convergence
        loss_iters = [h['iteration'] for h in self.loss_history]
        losses = [h['final_loss'] for h in self.loss_history]
        ax2.plot(loss_iters, losses, 'o-', linewidth=2)
        ax2.set_xlabel('Meta-iteration')
        ax2.set_ylabel('Final training loss')
        ax2.set_title('Learning convergence')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved evolution.png")

    def plot_layer_breakdown(self):
        """Plasticity by layer and module type."""
        final_iter = len(self.P_history)
        P = self._load_P(final_iter)

        # Organize by layer
        layer_data = {}
        for name, values in P.items():
            layer, module = self._parse_name(name)
            if layer is not None and module is not None:
                if layer not in layer_data:
                    layer_data[layer] = {'attn': [], 'mlp': [], 'ln': []}
                layer_data[layer][module].append(values.mean().item())

        layers = sorted(layer_data.keys())

        attn_means = [np.mean(layer_data[l]['attn']) if layer_data[l]['attn'] else 0
                      for l in layers]
        mlp_means = [np.mean(layer_data[l]['mlp']) if layer_data[l]['mlp'] else 0
                     for l in layers]

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(layers))
        width = 0.35

        ax.bar(x - width/2, attn_means, width, label='Attention')
        ax.bar(x + width/2, mlp_means, width, label='MLP')
        ax.axhline(y=1.0, color='red', linestyle=':', label='Uniform')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean P value')
        ax.set_title('Layer-wise plasticity')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'layer_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved layer_breakdown.png")

    def plot_module_comparison(self):
        """Compare attention vs MLP vs LayerNorm."""
        final_iter = len(self.P_history)
        P = self._load_P(final_iter)

        attn_vals = []
        mlp_vals = []
        ln_vals = []

        for name, values in P.items():
            layer, module = self._parse_name(name)
            vals = values.flatten().tolist()

            if module == 'attn':
                attn_vals.extend(vals)
            elif module == 'mlp':
                mlp_vals.extend(vals)
            elif module == 'ln':
                ln_vals.extend(vals)

        fig, ax = plt.subplots(figsize=(8, 5))

        data = [attn_vals, mlp_vals, ln_vals]
        labels = ['Attention', 'MLP', 'LayerNorm']

        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.axhline(y=1.0, color='red', linestyle=':', label='Uniform')
        ax.set_ylabel('P value')
        ax.set_title('Module type comparison')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Print statistics
        print("\nModule statistics:")
        for label, vals in zip(labels, data):
            print(f"  {label:12s}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'module_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved module_comparison.png")

    def plot_heatmap(self):
        """2D heatmap of P across layers and module components."""
        final_iter = len(self.P_history)
        P = self._load_P(final_iter)

        # Build matrix
        layer_modules = {}
        for name, values in P.items():
            layer, module = self._parse_name(name)
            if layer is not None and module is not None:
                key = (layer, module)
                if key not in layer_modules:
                    layer_modules[key] = []
                layer_modules[key].append(values.mean().item())

        # Get dimensions
        layers = sorted(set(k[0] for k in layer_modules.keys()))
        modules = ['attn', 'mlp', 'ln']

        matrix = np.zeros((len(layers), len(modules)))
        for i, layer in enumerate(layers):
            for j, module in enumerate(modules):
                key = (layer, module)
                if key in layer_modules:
                    matrix[i, j] = np.mean(layer_modules[key])

        fig, ax = plt.subplots(figsize=(6, max(6, len(layers) * 0.6)))

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)

        ax.set_xticks(range(len(modules)))
        ax.set_xticklabels(['Attention', 'MLP', 'LayerNorm'])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f'Layer {l}' for l in layers])

        ax.set_xlabel('Module type')
        ax.set_ylabel('Layer')
        ax.set_title('Plasticity heatmap')

        # Add values
        for i in range(len(layers)):
            for j in range(len(modules)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean P value')

        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap.png")

    def generate_all(self):
        """Generate all visualizations."""
        print("\nGenerating visualizations...")
        print("="*60)

        self.plot_evolution()
        self.plot_layer_breakdown()
        self.plot_module_comparison()
        self.plot_heatmap()

        print("="*60)
        print(f"All plots saved to: {self.experiment_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', help='Path to experiment directory')
    args = parser.parse_args()

    viz = PVisualizer(args.experiment_dir)
    viz.generate_all()


if __name__ == '__main__':
    main()
