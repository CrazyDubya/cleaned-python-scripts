"""
LoRA Fine-tuning for MLX Models

This script provides LoRA (Low-Rank Adaptation) fine-tuning capabilities for MLX models.
Supports Llama, Mixtral, and Phi2 models.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

# NOTE: You need to install mlx-examples and have the following imports available:
# from mlx_examples.lora import LoRALinear
# from mlx_examples.trainer import TrainingArgs, evaluate, train
# from mlx_examples.utils import generate, load

# Placeholder imports - replace with actual MLX imports
try:
    from mlx_examples.lora import LoRALinear
    from mlx_examples.trainer import TrainingArgs, evaluate, train
    from mlx_examples.utils import generate, load
    from mlx_examples.models import llama, mixtral, phi2
    SUPPORTED_MODELS = [llama.Model, mixtral.Model, phi2.Model]
except ImportError:
    print("Warning: MLX examples not found. Please install mlx-examples.")
    SUPPORTED_MODELS = []


class Dataset:
    """Light-weight wrapper to hold lines from a jsonl file"""

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for LoRA training"""
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    
    # Generation args
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


def load_dataset(args) -> Tuple[Dataset, Dataset, Dataset]:
    """Load training, validation, and test datasets"""
    names = ("train", "valid", "test")
    train, valid, test = (Dataset(Path(args.data) / f"{n}.jsonl") for n in names)
    
    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def main():
    """Main training loop"""
    if not SUPPORTED_MODELS:
        print("Error: MLX examples not properly installed. Cannot proceed.")
        sys.exit(1)
        
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load(args.model)

    if model.__class__ not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model.__class__} not supported. "
            f"Supported models: {SUPPORTED_MODELS}"
        )

    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args)

    # Resume training the given adapters
    if args.resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)
        
    # Initialize training args
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=args.adapter_file,
    )
    
    if args.train:
        print("Training")
        model.train()
        opt = optim.Adam(learning_rate=args.learning_rate)
        
        # Train model
        train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
        )

    # Load the LoRA adapter weights
    if not Path(args.adapter_file).is_file():
        raise ValueError(
            f"Adapter file {args.adapter_file} missing. "
            "Use --train to learn and save the adapters.npz."
        )
    model.load_weights(args.adapter_file, strict=False)

    if args.test:
        print("Testing")
        model.eval()

        test_loss = evaluate(
            model=model,
            dataset=test_set,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
        )

        test_ppl = math.exp(test_loss)
        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if args.prompt is not None:
        print("Generating")
        model.eval()
        generate(
            model=model,
            tokenizer=tokenizer,
            temp=args.temp,
            max_tokens=args.max_tokens,
            prompt=args.prompt,
            verbose=True,
        )


if __name__ == "__main__":
    main()