# Python Scripts Collection

A curated collection of Python scripts for machine learning, data processing, and utility tasks. All scripts have been cleaned of sensitive information and organized with proper documentation.

## 🗂️ Directory Structure

```
├── ai_tools/           # AI and machine learning utilities
├── data_processing/    # Data analysis and conversion tools  
├── mlx_lora/          # LoRA fine-tuning for MLX models
├── ollama_tools/      # Ollama integration scripts
├── utilities/         # General-purpose utilities
└── requirements.txt   # Python dependencies
```

## 🚀 Quick Start

1. **Clone or download this repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables** (see Security section below)
4. **Browse individual directories** for specific tools

## 📁 Categories

### 🤖 AI Tools (`ai_tools/`)
- **Grok Client:** X.AI API integration with retry logic
- **Actor Simulation:** Multi-persona AI simulation system
- **Image Generator:** Hyperbolic API image generation tool
- **Model Loader:** Universal loader for MLX, Transformers, Ollama

### 📊 Data Processing (`data_processing/`)  
- **JSON Text Processor:** Convert JSON transcripts to organized text
- **Conversation Analyzer:** Extract programming content from chat logs

### 🧠 MLX LoRA (`mlx_lora/`)
- **LoRA Trainer:** Fine-tune language models efficiently
- **LoRA Linear:** Low-rank adaptation layer implementation  
- **Training Utils:** Comprehensive training pipeline

### 🦙 Ollama Tools (`ollama_tools/`)
- **Text Refiner:** Multi-stage text improvement system

### 🛠️ Utilities (`utilities/`)
- **Mandelbrot Renderer:** Terminal-based fractal visualization
- **Directory Scanner:** Code analysis and file system exploration
- **XML Parser:** Robust parsing with fallback strategies

## 🔐 Security & Environment Setup

### Required Environment Variables

Create a `.env` file or set these variables:

```bash
# For AI tools
export XAI_API_KEY="your_xai_api_key_here"
export HYPERBOLIC_API_KEY="your_hyperbolic_api_key_here"

# Optional: for enhanced features
export OPENAI_API_KEY="your_openai_key_here"
export HF_TOKEN="your_huggingface_token_here"
```

### Security Notes
- ✅ All API keys have been removed from code
- ✅ Hardcoded file paths made configurable  
- ✅ Sensitive information sanitized
- ✅ Environment-based configuration

## 🏃‍♂️ Example Usage

### Quick AI Text Generation
```bash
cd ai_tools
python grok_client.py  # Requires XAI_API_KEY
```

### Data Analysis Pipeline
```bash
cd data_processing
python conversation_analyzer.py conversations.json --output-dir results/
```

### Visual Mandelbrot Exploration
```bash
cd utilities  
python mandelbrot.py --interactive
```

### Text Refinement with Ollama
```bash
cd ollama_tools
python text_refiner.py --topic "technical writing" --file document.txt
```

## 📦 Dependencies

### Core Requirements
- `numpy` - Numerical computing
- `requests` - HTTP requests
- `pathlib2` - Path handling (older Python)

### AI/ML Requirements  
- `transformers` - Hugging Face models
- `ollama` - Ollama integration
- `tenacity` - Retry logic
- `mlx` - Apple MLX framework (optional)

### Data Processing
- `textblob` - Text analysis
- `pandas` - Data manipulation

### Optional Enhancements
- `jupyter` - Notebook processing
- `yfinance` - Stock data
- `psutil` - System monitoring

## 🧪 Testing

Each script includes built-in testing capabilities:

```bash
# Test individual components
python ai_tools/model_loader.py --test
python utilities/xml_parser.py --verbose  
python data_processing/conversation_analyzer.py --summary-only
```

## 📖 Documentation

- Each directory contains a detailed `README.md`
- Scripts include `--help` for usage information
- Docstrings provide API documentation
- Comments explain complex logic

## 🔄 Development Workflow

1. **Explore:** Browse directories and READMEs
2. **Install:** Set up dependencies and environment
3. **Test:** Run scripts with sample data
4. **Customize:** Modify for your specific needs
5. **Integrate:** Combine tools for complex workflows

## 🤝 Contributing

This is a cleaned collection of utility scripts. To contribute:

1. Follow existing code style and documentation patterns
2. Remove any sensitive information before committing
3. Add appropriate error handling and logging
4. Update README files for new features

## 📝 License

Individual scripts may have their own licenses. Check file headers for details. Default is permissive open-source usage.

## ⚠️ Important Notes

- **API Costs:** Some tools use paid APIs (X.AI, Hyperbolic)
- **Model Storage:** MLX and large models require significant disk space
- **Performance:** Processing large datasets may require substantial RAM
- **Dependencies:** Some features require external services (Ollama, etc.)

---

**Happy coding! 🎉** Explore the directories for detailed documentation and examples.