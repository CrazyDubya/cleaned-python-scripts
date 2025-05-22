# Utilities

General-purpose utility scripts for various tasks.

## Scripts

### `mandelbrot.py`
Renders the Mandelbrot set in the terminal using ASCII characters.

**Features:**
- Auto-adapts to terminal size
- Configurable zoom and center point
- Interactive explorer mode
- Preset interesting locations

**Usage:**
```bash
# Basic rendering
python mandelbrot.py

# Custom parameters
python mandelbrot.py --zoom 100 --center-real -0.7 --center-imag 0.1 --iterations 200

# Interactive explorer
python mandelbrot.py --interactive
```

**Interactive commands:**
- `r` - render at current position
- `z <zoom>` - set zoom level  
- `c <real> <imag>` - set center point
- `p <preset>` - jump to preset location
- `q` - quit

### `directory_scanner.py`
Enhanced directory scanner for code analysis and file system exploration.

**Features:**
- Scans Python files, databases, notebooks, configs
- Extracts schema from SQLite databases
- Processes Jupyter notebook content
- Python code analysis (imports, functions, classes)
- Summary statistics by file type

**Usage:**
```bash
# Scan current directory
python directory_scanner.py

# Scan specific directory with output file
python directory_scanner.py /path/to/project --output scan_results.txt

# Show summary only  
python directory_scanner.py /path/to/project --summary-only

# Limit number of files processed
python directory_scanner.py /path/to/project --max-files 50
```

### `xml_parser.py`
Robust XML parser with multiple fallback strategies for handling malformed XML.

**Features:**
- Multiple parsing strategies
- Handles malformed XML and encoding issues
- XML structure salvage capabilities
- Converts XML to dictionary format
- Detailed parsing attempt reporting

**Usage:**
```bash
# Parse XML file
python xml_parser.py document.xml --verbose

# Parse XML text directly
python xml_parser.py --text "<root><item>test</item></root>"

# Convert to dictionary format
python xml_parser.py document.xml --to-dict

# Verbose parsing with attempt reporting
python xml_parser.py malformed.xml --verbose
```

**Parsing strategies:**
1. Direct parsing with cleanup
2. Root element wrapping
3. Common issue fixing
4. XML declaration extraction
5. Structure salvage
6. Minimal structure creation

## Supported File Types

### Directory Scanner
- **Python files:** `.py` - extracts imports, functions, classes
- **Databases:** `.db` - extracts schema and table info
- **Notebooks:** `.ipynb` - extracts cells and content  
- **Config files:** `.ini`, `.cfg`, `.conf`, `.env`, `.toml`, `.yaml`
- **Documentation:** `.md`, `.txt`, `README`, `LICENSE`

### XML Parser
- Handles any XML content with various malformation issues
- Supports XML declarations, namespaces, attributes
- Can salvage partial or broken XML structures

## Dependencies

```bash
pip install pathlib2  # For older Python versions
```

All utilities use only standard library modules where possible.