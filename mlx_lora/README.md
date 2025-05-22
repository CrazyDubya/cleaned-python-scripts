# MLX LoRA Fine-tuning

This directory contains LoRA (Low-Rank Adaptation) fine-tuning implementations for MLX models.

## Files

- `lora_trainer.py` - Main training script with CLI interface
- `lora_linear.py` - LoRA linear layer implementation
- `trainer.py` - Training utilities and data classes

## Prerequisites

1. Install MLX and MLX Examples:
   ```bash
   pip install mlx
   git clone https://github.com/ml-explore/mlx-examples.git
   cd mlx-examples
   pip install -e .
   ```

2. Prepare your dataset in JSONL format with train, valid, and test splits.

## Usage

### Basic Training

```bash
python lora_trainer.py --train --data /path/to/data --model /path/to/model
```

### Generation

```bash
python lora_trainer.py --prompt "Your prompt here" --adapter-file adapters.npz
```

### Full Example

```bash
# Train for 1000 iterations with custom settings
python lora_trainer.py \
    --train \
    --data ./data \
    --model mlx-community/phi-2 \
    --lora-layers 16 \
    --batch-size 4 \
    --iters 1000 \
    --learning-rate 1e-5 \
    --adapter-file my_adapters.npz

# Generate with trained adapter
python lora_trainer.py \
    --prompt "Explain quantum computing" \
    --adapter-file my_adapters.npz \
    --max-tokens 200
```

## Arguments

- `--model`: Model path or Hugging Face repo
- `--data`: Directory with train/valid/test.jsonl files
- `--lora-layers`: Number of layers to fine-tune (default: 16)
- `--batch-size`: Training batch size (default: 4)
- `--iters`: Number of training iterations (default: 1000)
- `--learning-rate`: Adam learning rate (default: 1e-5)
- `--adapter-file`: Path to save/load adapter weights
- `--prompt`: Text prompt for generation
- `--max-tokens`: Maximum tokens to generate

## Data Format

Your JSONL files should contain objects with a "text" field:

```json
{"text": "Your training text here"}
{"text": "Another training example"}
```

## Supported Models

- Llama models
- Mixtral models  
- Phi2 models

## Notes

- Adapters are saved automatically during training
- Use `--resume-adapter-file` to continue training from existing adapters
- The script will validate your data before training