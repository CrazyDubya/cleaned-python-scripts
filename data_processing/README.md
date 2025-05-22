# Data Processing Tools

Scripts for processing and analyzing various data formats.

## Scripts

### `json_text_processor.py`
Processes JSON files containing text data and extracts specific content.

**Features:**
- Converts JSON to text with datetime headers
- Organizes files by type into subdirectories  
- Extracts statements from specific speakers
- Configurable year filtering
- Processing statistics

**Usage:**
```bash
# Process all files in a directory
python json_text_processor.py /path/to/data --speaker "SENATOR LANZA"

# Just show statistics
python json_text_processor.py /path/to/data --stats-only

# Skip certain years
python json_text_processor.py /path/to/data --skip-years 2008 2009
```

### `conversation_analyzer.py`  
Analyzes conversation JSON files for programming content.

**Features:**
- Extracts programming language mentions
- Identifies code blocks by language
- Parses Python functions from code
- Generates CSV reports
- Summary statistics

**Usage:**
```bash
# Analyze conversations and save CSV reports
python conversation_analyzer.py conversations.json --output-dir results/

# Show summary only
python conversation_analyzer.py conversations.json --summary-only
```

**Output files:**
- `prog_lang_mentions.csv` - Language mention counts
- `code_block_counts.csv` - Code block counts by language  
- `function_counts.csv` - Python function occurrences

## Data Formats

### JSON Text Processor
Expects JSON files with structure:
```json
{
  "result": {
    "dateTime": "2023-01-01 12:00:00",
    "text": "Meeting transcript with speaker names..."
  }
}
```

### Conversation Analyzer
Expects conversation data with structure:
```json
[
  {
    "mapping": {
      "node_id": {
        "message": {
          "author": {"role": "user"},
          "content": {"parts": ["Message text here"]}
        }
      }
    }
  }
]
```

## Dependencies

```bash
pip install textblob pandas
```