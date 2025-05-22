# Ollama Tools

Scripts for working with Ollama language models.

## Scripts

### `text_refiner.py`
Iterative text refinement using Ollama models.

**Features:**
- Multi-stage refinement process
- Multiple parsing strategies for LLM responses
- Context preservation across refinement passes
- Configurable refinement stages
- Statistics tracking
- Interactive and batch modes

**Usage:**
```bash
# Refine text from file
python text_refiner.py --topic "technical writing" --file document.txt

# Refine text directly
python text_refiner.py --topic "creative writing" --text "Your text here"

# Interactive mode
python text_refiner.py --topic "editing" --interactive

# Custom model and output
python text_refiner.py --topic "research" --file paper.txt --model llama2 --output refined_paper.txt

# Custom refinement stages
python text_refiner.py --topic "content" --text "text" --stages '[[75, 2], [25, 5], [10, 10]]'
```

## Refinement Process

The tool uses a multi-stage approach:

1. **Stage 1:** Broad refinement (50% of text, 2 passes)
2. **Stage 2:** Medium refinement (25% of text, 4 passes)  
3. **Stage 3:** Fine refinement (10% of text, 10 passes)
4. **Stage 4:** Detail refinement (5% of text, 20 passes)

Each stage focuses on progressively smaller portions with more passes for detailed improvement.

## Response Parsing

The tool handles various LLM response formats:
- JSON arrays
- Comma-separated lists
- Newline-separated items
- Numbered lists (1. item, 2. item)
- Bullet points (- item, * item)
- Semicolon-separated items

## Arguments

- `--topic`: Topic/context for refinement (required)
- `--text`: Text to refine directly
- `--file`: File containing text to refine
- `--model`: Ollama model to use (default: gemma2:2b)
- `--output`: Output filename for results
- `--stages`: Custom refinement stages as JSON array
- `--interactive`: Interactive input mode

## Custom Stages Format

Stages are defined as `[percentage, passes]` pairs:

```json
[
  [50, 2],    // Refine 50% in 2 passes
  [25, 4],    // Refine 25% in 4 passes  
  [10, 10],   // Refine 10% in 10 passes
  [5, 20]     // Refine 5% in 20 passes
]
```

## Requirements

- Ollama installed and running
- Models pulled locally (e.g., `ollama pull gemma2:2b`)

```bash
pip install ollama
```

## Available Models

Common Ollama models for text refinement:
- `gemma2:2b` - Fast, lightweight
- `llama2` - Good balance of speed/quality
- `mistral` - Strong instruction following
- `codellama` - Good for technical content
- `neural-chat` - Conversational refinement

Pull models with: `ollama pull <model-name>`