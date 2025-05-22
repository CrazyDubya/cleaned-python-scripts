# AI Tools

Collection of AI and machine learning utility scripts.

## Scripts

### `grok_client.py`
Client for interacting with Grok (X.AI) API.
- Supports deferred completions
- Retry logic for API calls
- Simple prompt interface

**Requirements:** `XAI_API_KEY` environment variable

```bash
python grok_client.py
```

### `actor_simulation.py`  
AI actor simulation using Grok API.
- Multiple AI personas with different archetypes
- Emotional state tracking
- Memory and context management
- Simulation logging

```bash
python actor_simulation.py
```

### `image_generator.py`
Image generation using Hyperbolic API.
- Interactive menu interface
- Generation history tracking
- Multiple model support

**Requirements:** `HYPERBOLIC_API_KEY` environment variable

```bash
python image_generator.py
```

### `model_loader.py`
Universal model loader for different ML frameworks.
- MLX model support
- Hugging Face Transformers
- Ollama integration
- Interactive testing

```bash
python model_loader.py --framework transformers --model gpt2 --test
```

## Environment Variables

Create a `.env` file or set these in your environment:

```bash
export XAI_API_KEY="your_xai_api_key_here"
export HYPERBOLIC_API_KEY="your_hyperbolic_api_key_here"
```

## Dependencies

```bash
pip install requests tenacity ollama transformers
```