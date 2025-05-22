"""
Enhanced Directory Scanner

Scans directories and creates representations of files and their contents.
Supports Python files, databases, Jupyter notebooks, and configuration files.
"""

import os
import sqlite3
import json
import argparse
from pathlib import Path
from typing import List, Set, Dict, Any, Optional


class DirectoryScanner:
    """Enhanced directory scanner for code analysis"""
    
    def __init__(self, start_path: str = "."):
        self.start_path = Path(start_path)
        self.supported_extensions = {
            '.py', '.db', '.ipynb', '.txt', '.md', '.json', '.yaml', '.yml', 
            '.toml', '.ini', '.cfg', '.conf', '.env'
        }
        self.supported_files = {
            'requirements.txt', 'config.ini', 'setup.py', 'setup.cfg',
            'pyproject.toml', 'Dockerfile', 'docker-compose.yml',
            'README.md', 'LICENSE', 'Makefile'
        }
        
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed"""
        return (
            file_path.suffix.lower() in self.supported_extensions or
            file_path.name in self.supported_files
        )
    
    def process_sqlite_database(self, db_path: Path) -> List[str]:
        """Extract schema information from SQLite database"""
        output = []
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                output.append(f"Database: {db_path.name}")
                output.append(f"Tables: {len(tables)}")
                
                for table in tables:
                    table_name = table[0]
                    output.append(f"\nTable: {table_name}")
                    
                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    
                    for col in columns:
                        output.append(f"  {col[1]} {col[2]}")
                        
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    count = cursor.fetchone()[0]
                    output.append(f"  Rows: {count}")
                    
        except Exception as e:
            output.append(f"Error reading database: {e}")
            
        return output
    
    def process_jupyter_notebook(self, nb_path: Path) -> List[str]:
        """Extract content from Jupyter notebook"""
        output = []
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
                
            output.append(f"Notebook: {nb_path.name}")
            
            cells = notebook.get('cells', [])
            markdown_cells = 0
            code_cells = 0
            
            for i, cell in enumerate(cells):
                cell_type = cell.get('cell_type', 'unknown')
                source = cell.get('source', [])
                
                if cell_type == 'markdown':
                    markdown_cells += 1
                    if source:
                        output.append(f"\n--- Markdown Cell {i+1} ---")
                        output.extend(source)
                        
                elif cell_type == 'code':
                    code_cells += 1
                    if source:
                        output.append(f"\n--- Code Cell {i+1} ---")
                        output.extend(source)
                        
            output.insert(1, f"Cells: {len(cells)} ({code_cells} code, {markdown_cells} markdown)")
            
        except Exception as e:
            output.append(f"Error reading notebook: {e}")
            
        return output
    
    def process_text_file(self, file_path: Path) -> List[str]:
        """Process text-based files"""
        output = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            output.append(f"File: {file_path.name}")
            output.append(f"Lines: {len(lines)}")
            output.append(f"Characters: {len(content)}")
            
            # For Python files, extract basic info
            if file_path.suffix == '.py':
                output.extend(self._analyze_python_file(content))
                
            # Show first few lines for small files
            if len(lines) <= 50:
                output.append("\nContent:")
                output.extend(lines)
            else:
                output.append("\nFirst 20 lines:")
                output.extend(lines[:20])
                output.append("...")
                output.append("Last 10 lines:")
                output.extend(lines[-10:])
                
        except Exception as e:
            output.append(f"Error reading file: {e}")
            
        return output
    
    def _analyze_python_file(self, content: str) -> List[str]:
        """Analyze Python file content"""
        analysis = []
        lines = content.split('\n')
        
        # Count various elements
        imports = [line for line in lines if line.strip().startswith(('import ', 'from '))]
        functions = [line for line in lines if line.strip().startswith('def ')]
        classes = [line for line in lines if line.strip().startswith('class ')]
        
        if imports:
            analysis.append(f"Imports: {len(imports)}")
        if functions:
            analysis.append(f"Functions: {len(functions)}")
        if classes:
            analysis.append(f"Classes: {len(classes)}")
            
        # Extract docstring if present
        if content.strip().startswith('"""') or content.strip().startswith("'''"):
            end_quote = '"""' if content.strip().startswith('"""') else "'''"
            try:
                docstring_end = content.find(end_quote, 3)
                if docstring_end > 0:
                    docstring = content[3:docstring_end].strip()
                    analysis.append(f"Docstring: {docstring[:100]}...")
            except:
                pass
                
        return analysis
    
    def scan_directory(self, max_files: int = 100) -> str:
        """Scan directory and create representation of relevant files"""
        output = []
        files_processed = 0
        
        output.append(f"Directory Scan: {self.start_path.absolute()}")
        output.append("=" * 80)
        
        for root, dirs, files in os.walk(self.start_path):
            # Skip common irrelevant directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                '__pycache__', 'node_modules', '.git', '.vscode', '.idea', 
                'venv', 'env', '.env', 'build', 'dist'
            }]
            
            root_path = Path(root)
            
            for file in files:
                if files_processed >= max_files:
                    output.append(f"\n[Truncated - processed {max_files} files maximum]")
                    break
                    
                file_path = root_path / file
                
                if not self.should_process_file(file_path):
                    continue
                    
                output.append(f"\n{'='*80}")
                output.append(f"FILE: {file_path.relative_to(self.start_path)}")
                output.append('='*80)
                
                # Process based on file type
                if file_path.suffix == '.db':
                    output.extend(self.process_sqlite_database(file_path))
                elif file_path.suffix == '.ipynb':
                    output.extend(self.process_jupyter_notebook(file_path))
                else:
                    output.extend(self.process_text_file(file_path))
                    
                files_processed += 1
                
            if files_processed >= max_files:
                break
        
        output.append(f"\n\nScan completed. Processed {files_processed} files.")
        return '\n'.join(output)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the directory"""
        summary = {
            'total_files': 0,
            'by_extension': {},
            'by_type': {
                'python': 0,
                'notebooks': 0,
                'databases': 0,
                'config': 0,
                'documentation': 0,
                'other': 0
            }
        }
        
        for root, dirs, files in os.walk(self.start_path):
            # Skip irrelevant directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                '__pycache__', 'node_modules', '.git', '.vscode', '.idea', 
                'venv', 'env', '.env', 'build', 'dist'
            }]
            
            for file in files:
                file_path = Path(root) / file
                
                if not self.should_process_file(file_path):
                    continue
                    
                summary['total_files'] += 1
                
                ext = file_path.suffix.lower()
                summary['by_extension'][ext] = summary['by_extension'].get(ext, 0) + 1
                
                # Categorize
                if ext == '.py':
                    summary['by_type']['python'] += 1
                elif ext == '.ipynb':
                    summary['by_type']['notebooks'] += 1
                elif ext == '.db':
                    summary['by_type']['databases'] += 1
                elif ext in {'.ini', '.cfg', '.conf', '.env', '.toml', '.yaml', '.yml'}:
                    summary['by_type']['config'] += 1
                elif ext in {'.md', '.txt'} or file_path.name in {'README', 'LICENSE'}:
                    summary['by_type']['documentation'] += 1
                else:
                    summary['by_type']['other'] += 1
                    
        return summary


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Enhanced directory scanner")
    parser.add_argument(
        "directory", 
        nargs='?', 
        default=".",
        help="Directory to scan (default: current directory)"
    )
    parser.add_argument(
        "--max-files", 
        type=int, 
        default=100,
        help="Maximum number of files to process (default: 100)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only summary statistics"
    )
    parser.add_argument(
        "--output",
        help="Output file to save results"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return
        
    scanner = DirectoryScanner(args.directory)
    
    if args.summary_only:
        summary = scanner.get_summary()
        print("\nDirectory Summary:")
        print(f"Total relevant files: {summary['total_files']}")
        print(f"Python files: {summary['by_type']['python']}")
        print(f"Notebooks: {summary['by_type']['notebooks']}")
        print(f"Databases: {summary['by_type']['databases']}")
        print(f"Config files: {summary['by_type']['config']}")
        print(f"Documentation: {summary['by_type']['documentation']}")
        print(f"Other: {summary['by_type']['other']}")
        
        if summary['by_extension']:
            print(f"\nBy extension:")
            for ext, count in sorted(summary['by_extension'].items()):
                print(f"  {ext or 'no extension'}: {count}")
    else:
        result = scanner.scan_directory(args.max_files)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result)
            print(f"Results saved to {args.output}")
        else:
            print(result)


if __name__ == "__main__":
    main()