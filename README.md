# Python Scripts Collection

This repository contains a collection of Python scripts for various purposes including machine learning, data processing, and utilities.

## Scripts Overview

### Machine Learning & AI
- **`mlx_lora/`** - LoRA fine-tuning implementation for MLX
- **`model_loader.py`** - Universal model loader for MLX and Hugging Face
- **`ai_tools/`** - Various AI utilities and conversation managers
- **`ollama_tools/`** - Ollama integration scripts

### Data Processing
- **`data_processing/`** - Scripts for text analysis, JSON processing, and data conversion
- **`xml_parser.py`** - Robust XML parsing utility

### Utilities
- **`mandelbrot.py`** - Mandelbrot set visualization
- **`directory_scanner.py`** - Enhanced file system scanner

### GUI Tools
- **`vision_analysis/`** - Vision model comparison tools (fragments)

## Security Notes

- All API keys have been removed and must be set via environment variables
- Hardcoded file paths have been made configurable
- Sensitive information has been sanitized

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set required environment variables:
   ```bash
   export XAI_API_KEY="your_xai_api_key_here"
   export HYPERBOLIC_API_KEY="your_hyperbolic_api_key_here"
   ```

## Usage

Each script directory contains its own README with specific usage instructions.

## License

Individual scripts may have their own licenses. Please check file headers for details.