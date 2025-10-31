#!/usr/bin/env python3
"""
Strips embeddings from a trained GPT2 model checkpoint.

Usage:
    python strip_embeddings.py model.pt stripped_model.pt

Reinitializes the embedding layers (wte, wpe, lm_head) while keeping
all the transformer block weights intact. Not sure if we should be wiping wpe but easy to comment out. 
"""

import torch
import torch.nn as nn
import sys
import os


def strip_embeddings_from_checkpoint(input_path, output_path):
    """Load checkpoint, strip embeddings, save result."""
    print(f"Loading model from: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Model checkpoint not found: {input_path}")

    checkpoint = torch.load(input_path, map_location='cpu')

    # Extract state dict and config from checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', None)
            extra_info = {k: v for k, v in checkpoint.items()
                         if k not in ['model_state_dict', 'config']}
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            config = checkpoint.get('config', None)
            extra_info = {k: v for k, v in checkpoint.items()
                         if k not in ['state_dict', 'config']}
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            config = checkpoint.get('config', None)
            extra_info = {k: v for k, v in checkpoint.items() if k not in ['model', 'config']}

        else:
            # Just a raw state dict
            state_dict = checkpoint
            config = None
            extra_info = {}
    else:
        raise ValueError("Unexpected checkpoint format")

    print(f"Original model has {len(state_dict)} parameters")

    # Figure out config from state dict if needed
    if config is None:
        config = extract_config_from_state_dict(state_dict)
    else: 
        inferred = extract_config_from_state_dict(state_dict)
        for k, v in inferred.items():
            config.setdefault(k, v)

    print(f"\nModel config:")
    for key in ['vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd']: 
        value = config.get(key, '<unknown>')
        print(f"{key}: {value}")

    # Do the stripping
    print("\nStripping embeddings...")
    stripped_state_dict = strip_embedding_layers(state_dict, config)

    print(f"Stripped model has {len(stripped_state_dict)} parameters")

    # Show what changed
    original_params = sum(p.numel() for p in state_dict.values())
    stripped_params = sum(p.numel() for p in stripped_state_dict.values())
    print(f"\nParameter count:")
    print(f"  Original: {original_params:,}")
    print(f"  Stripped: {stripped_params:,}")
    print(f"  Removed: {original_params - stripped_params:,}")

    # Save it
    output_checkpoint = {
        'model': stripped_state_dict,
        'config': config,
        **extra_info
    }

    print(f"\nSaving stripped model to: {output_path}")
    torch.save(output_checkpoint, output_path)
    print("Done!")


def extract_config_from_state_dict(state_dict):
    """Figure out model config by looking at the state dict."""
    # Get vocab size and embedding dim from wte
    if 'wte.weight' in state_dict:
        vocab_size, n_embd = state_dict['wte.weight'].shape
    elif 'transformer.wte.weight' in state_dict:
        vocab_size, n_embd = state_dict['transformer.wte.weight'].shape
    else:
        raise ValueError("Can't find embedding weights in state dict")

    # Get block size from wpe
    if 'wpe.weight' in state_dict:
        block_size = state_dict['wpe.weight'].shape[0]
    elif 'transformer.wpe.weight' in state_dict:
        block_size = state_dict['transformer.wpe.weight'].shape[0]
    else:
        raise ValueError("Can't find position embedding weights in state dict")

    # Count the transformer layers
    n_layer = 0
    for key in state_dict.keys():
        if 'blocks.' in key or 'h.' in key:
            if 'blocks.' in key:
                layer_num = int(key.split('blocks.')[1].split('.')[0])
            else:  # HuggingFace style uses 'h.'
                layer_num = int(key.split('h.')[1].split('.')[0])
            n_layer = max(n_layer, layer_num + 1)

    # Try to infer n_head from common configs
    common_configs = [
        (384, 6), (256, 4), (512, 8), (768, 12), (1024, 16), (1280, 20), (1600, 25)
    ]
    n_head = None
    for embd, head in common_configs:
        if embd == n_embd:
            n_head = head
            break

    if n_head is None:
        # Assume head_dim = 64
        n_head = n_embd // 64
        if n_head == 0:
            n_head = 1

    return {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd
    }


def strip_embedding_layers(state_dict, config):
    """
    Reinitialize embedding layers while keeping transformer blocks.
    Weight tying gets handled when you load the model.
    """
    stripped_dict = {}
    std_emb = 0.02  # GPT-2 default

    stripped_keys = []
    kept_keys = []

    for key, value in state_dict.items():
        # Reinitialize these layers
        if any(layer in key for layer in ['wte', 'wpe', 'lm_head']):
            print(f"  Reinitializing: {key} {list(value.shape)}")
            new_param = torch.randn_like(value) * std_emb
            stripped_dict[key] = new_param
            stripped_keys.append(key)
        else:
            # Keep everything else (transformer blocks, layer norms)
            stripped_dict[key] = value.clone()
            kept_keys.append(key)

    print(f"\n  Stripped {len(stripped_keys)} layers")
    print(f"  Kept {len(kept_keys)} layers (transformer core)")

    return stripped_dict


def main():
    if len(sys.argv) < 2:
        print("Usage: python strip_embeddings_standalone.py <input_model.pt> [output_model.pt]")
        print("\nExample:")
        print("  python strip_embeddings_standalone.py trained_model.pt stripped_model.pt")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Default output name
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_stripped{ext}"

    strip_embeddings_from_checkpoint(input_path, output_path)


if __name__ == "__main__":
    main()
