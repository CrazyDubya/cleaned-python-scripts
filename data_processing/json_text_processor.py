"""
JSON to Text Processing Tool

This script processes JSON files containing text data, cleans them,
and extracts specific content (like Senator Lanza statements).
"""

import os
import json
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


class JSONTextProcessor:
    """Processes JSON files and extracts text content"""
    
    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory)
        self.line_number_pattern = re.compile(r'^\s*\d+\s+')
        
    def json_to_text(self, skip_years: Optional[List[str]] = None) -> None:
        """Convert JSON files to text files with datetime headers"""
        skip_years = skip_years or ['2008', '2009']
        
        print("Converting JSON files to text...")
        
        for year_dir in self.base_directory.iterdir():
            if not year_dir.is_dir() or year_dir.name in skip_years:
                continue
                
            print(f"Processing year: {year_dir.name}")
            
            for json_file in year_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract datetime and text
                    date_time = data.get('result', {}).get('dateTime', '')
                    text = data.get('result', {}).get('text', '')
                    
                    # Clean text by removing line numbers
                    lines = text.split('\n')
                    cleaned_lines = [
                        self.line_number_pattern.sub('', line.strip()) 
                        for line in lines
                    ]
                    cleaned_text = '\n'.join(cleaned_lines)
                    
                    # Create output file
                    txt_file = json_file.with_suffix('.txt')
                    with open(txt_file, 'w') as f:
                        f.write(f'DateTime: {date_time}\n\n{cleaned_text}\n')
                        
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    
        print("JSON to text conversion complete.")

    def organize_files_by_type(self, skip_years: Optional[List[str]] = None) -> None:
        """Organize files into subdirectories by type"""
        skip_years = skip_years or ['2008', '2009']
        
        print("Organizing files by type...")
        
        for year_dir in self.base_directory.iterdir():
            if not year_dir.is_dir() or year_dir.name in skip_years:
                continue
                
            print(f"Organizing files in: {year_dir.name}")
            
            # Create subdirectories
            txt_dir = year_dir / 'txt'
            json_dir = year_dir / 'json'
            txt_dir.mkdir(exist_ok=True)
            json_dir.mkdir(exist_ok=True)
            
            # Move files
            for file_path in year_dir.iterdir():
                if file_path.is_file():
                    if file_path.suffix == '.json':
                        shutil.move(str(file_path), str(json_dir / file_path.name))
                    elif file_path.suffix == '.txt':
                        shutil.move(str(file_path), str(txt_dir / file_path.name))
                        
        print("File organization complete.")

    def extract_speaker_statements(self, speaker_name: str = "SENATOR LANZA", 
                                 skip_years: Optional[List[str]] = None) -> None:
        """Extract all statements from a specific speaker"""
        skip_years = skip_years or ['2008', '2009']
        
        # Pattern to match speaker statements
        pattern = re.compile(
            rf'({re.escape(speaker_name)}:.*?)(?=( [A-Z\s]+:|$))', 
            re.DOTALL
        )
        
        print(f"Extracting statements from {speaker_name}...")
        
        for year_dir in self.base_directory.iterdir():
            if not year_dir.is_dir() or year_dir.name in skip_years:
                continue
                
            print(f"Processing year: {year_dir.name}")
            
            txt_dir = year_dir / 'txt'
            if not txt_dir.exists():
                print(f"No txt directory found in {year_dir.name}, skipping...")
                continue
                
            output_file = year_dir / f'all_{speaker_name.lower().replace(" ", "_")}_statements.txt'
            
            with open(output_file, 'w') as outf:
                for txt_file in txt_dir.glob("*.txt"):
                    try:
                        with open(txt_file, 'r') as inf:
                            text = inf.read()
                            
                        statements = pattern.findall(text)
                        for statement in statements:
                            # statement[0] contains the actual statement text
                            outf.write(statement[0].strip() + '\n\n')
                            
                    except Exception as e:
                        print(f"Error processing {txt_file}: {e}")
                        
        print(f"Speaker statement extraction complete.")

    def process_all(self, speaker_name: str = "SENATOR LANZA", 
                   skip_years: Optional[List[str]] = None) -> None:
        """Run complete processing pipeline"""
        print("Starting complete JSON text processing pipeline...")
        
        self.json_to_text(skip_years)
        self.organize_files_by_type(skip_years)
        self.extract_speaker_statements(speaker_name, skip_years)
        
        print("Processing pipeline complete!")

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            "total_years": 0,
            "total_json_files": 0,
            "total_txt_files": 0,
            "years_processed": []
        }
        
        for year_dir in self.base_directory.iterdir():
            if year_dir.is_dir():
                stats["total_years"] += 1
                stats["years_processed"].append(year_dir.name)
                
                # Count files
                json_count = len(list(year_dir.rglob("*.json")))
                txt_count = len(list(year_dir.rglob("*.txt")))
                
                stats["total_json_files"] += json_count
                stats["total_txt_files"] += txt_count
                
        return stats


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Process JSON files and extract text content"
    )
    parser.add_argument(
        "directory", 
        help="Base directory containing year subdirectories with JSON files"
    )
    parser.add_argument(
        "--speaker", 
        default="SENATOR LANZA",
        help="Speaker name to extract statements from"
    )
    parser.add_argument(
        "--skip-years",
        nargs="*",
        default=["2008", "2009"],
        help="Years to skip during processing"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't process files"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return
        
    processor = JSONTextProcessor(args.directory)
    
    if args.stats_only:
        stats = processor.get_statistics()
        print("\nProcessing Statistics:")
        print(f"Total years: {stats['total_years']}")
        print(f"Total JSON files: {stats['total_json_files']}")
        print(f"Total TXT files: {stats['total_txt_files']}")
        print(f"Years found: {', '.join(stats['years_processed'])}")
    else:
        processor.process_all(args.speaker, args.skip_years)


if __name__ == "__main__":
    main()