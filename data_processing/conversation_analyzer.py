"""
Conversation Analysis Tool

Analyzes conversation JSON files to extract programming language mentions,
code blocks, and function definitions.
"""

import json
import re
import ast
import csv
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any
from textblob import TextBlob


class ConversationAnalyzer:
    """Analyzes conversation data for programming content"""
    
    def __init__(self, conversations_file: str):
        self.conversations_file = Path(conversations_file)
        self.conversations = []
        self.load_conversations()
        
    def load_conversations(self) -> None:
        """Load conversations from JSON file"""
        if not self.conversations_file.exists():
            raise FileNotFoundError(f"Conversations file not found: {self.conversations_file}")
            
        with open(self.conversations_file, 'r') as f:
            conversations_data = json.load(f)
        
        # Extract messages from the 'mapping' key
        self.conversations = []
        
        for conversation in conversations_data:
            conversation_messages = []
            mapping = conversation.get('mapping', {})
            
            for node in mapping.values():
                message = node.get('message')
                if message and message.get('content'):
                    author_role = message['author']['role']
                    message_content = message['content'].get('parts', [])
                    conversation_messages.append({
                        'role': author_role,
                        'content': ' '.join(message_content)
                    })
            self.conversations.append(conversation_messages)

    def extract_code_blocks(self, message: str) -> List[str]:
        """Extract code blocks from a message"""
        return re.findall(r'```(.*?)```', message, re.DOTALL)

    def identify_language(self, code_block: str) -> str:
        """Identify the programming language of a code block"""
        first_line = code_block.strip().split('\n')[0].lower()
        
        language_indicators = {
            'python': ['python', 'py'],
            'bash': ['bash', 'shell', 'sh'],
            'javascript': ['javascript', 'js', 'node'],
            'markdown': ['markdown', 'md'],
            'sql': ['sql'],
            'json': ['json'],
            'yaml': ['yaml', 'yml'],
            'xml': ['xml'],
            'html': ['html'],
            'css': ['css']
        }
        
        for language, indicators in language_indicators.items():
            if any(indicator in first_line for indicator in indicators):
                return language
                
        return 'other'

    def extract_functions(self, code_block: str) -> List[str]:
        """Extract function names from Python code blocks"""
        try:
            tree = ast.parse(code_block)
            return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        except SyntaxError:
            return []

    def analyze_programming_content(self) -> Tuple[Counter, Counter, Counter]:
        """Analyze conversations for programming content"""
        
        # Programming languages and frameworks to look for
        tech_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 
            'go', 'kotlin', 'rust', 'scala', 'haskell', 'dart', 'lua', 'perl', 
            'r', 'julia', 'elixir', 'clojure', 'erlang', 'f#', 'groovy', 'lisp', 
            'matlab', 'objective-c', 'powershell', 'shell', 'sql', 'typescript', 
            'vb.net', 'html', 'css', 'xml', 'json', 'yaml', 'toml', 'ini',
            'markdown', 'latex', 'restructuredtext', 'django', 'flask', 'fastapi', 
            'express', 'react', 'angular', 'vue', 'spring', 'rails', 'laravel', '.net'
        ]
        
        prog_lang_mentions = Counter()
        code_block_counts = Counter()
        function_counts = Counter()

        for idx, conversation in enumerate(self.conversations):
            for message in conversation:
                content = message['content']

                # Count mentions of programming languages and frameworks
                content_lower = content.lower()
                for keyword in tech_keywords:
                    if keyword in content_lower:
                        prog_lang_mentions[keyword] += 1

                # Analyze code blocks
                code_blocks = self.extract_code_blocks(content)
                for code_block in code_blocks:
                    language = self.identify_language(code_block)
                    code_block_counts[language] += 1

                    # Extract functions from Python code blocks
                    if language == 'python':
                        functions = self.extract_functions(code_block)
                        for function in functions:
                            function_counts[(function, f"conversation_{idx}")] += 1

        return prog_lang_mentions, code_block_counts, function_counts

    def save_analysis_results(self, output_dir: str = ".") -> None:
        """Save analysis results to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        prog_lang_mentions, code_block_counts, function_counts = self.analyze_programming_content()

        # Save programming language mentions
        with open(output_path / 'prog_lang_mentions.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Language', 'Mentions'])
            writer.writerows(prog_lang_mentions.most_common())

        # Save code block counts
        with open(output_path / 'code_block_counts.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Language', 'Count'])
            writer.writerows(code_block_counts.most_common())

        # Save function counts
        with open(output_path / 'function_counts.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Function', 'Conversation', 'Count'])
            for (function, conversation), count in function_counts.items():
                writer.writerow([function, conversation, count])

        print(f"Analysis results saved to {output_path}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the conversations"""
        prog_lang_mentions, code_block_counts, function_counts = self.analyze_programming_content()
        
        total_messages = sum(len(conv) for conv in self.conversations)
        messages_with_code = 0
        
        for conversation in self.conversations:
            for message in conversation:
                if self.extract_code_blocks(message['content']):
                    messages_with_code += 1
        
        return {
            'total_conversations': len(self.conversations),
            'total_messages': total_messages,
            'messages_with_code': messages_with_code,
            'unique_languages_mentioned': len(prog_lang_mentions),
            'total_code_blocks': sum(code_block_counts.values()),
            'unique_functions_found': len(function_counts),
            'top_languages': prog_lang_mentions.most_common(5),
            'top_code_block_types': code_block_counts.most_common(5)
        }

    def print_summary(self) -> None:
        """Print a summary of the analysis"""
        stats = self.get_summary_statistics()
        
        print("\n" + "="*50)
        print("CONVERSATION ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total Conversations: {stats['total_conversations']}")
        print(f"Total Messages: {stats['total_messages']}")
        print(f"Messages with Code: {stats['messages_with_code']}")
        print(f"Unique Languages Mentioned: {stats['unique_languages_mentioned']}")
        print(f"Total Code Blocks: {stats['total_code_blocks']}")
        print(f"Unique Functions Found: {stats['unique_functions_found']}")
        
        print(f"\nTop Languages Mentioned:")
        for lang, count in stats['top_languages']:
            print(f"  {lang}: {count}")
            
        print(f"\nTop Code Block Types:")
        for lang, count in stats['top_code_block_types']:
            print(f"  {lang}: {count}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Analyze conversation data for programming content"
    )
    parser.add_argument(
        "conversations_file",
        help="Path to conversations JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for CSV files (default: current directory)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary, don't save CSV files"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = ConversationAnalyzer(args.conversations_file)
        
        if args.summary_only:
            analyzer.print_summary()
        else:
            analyzer.save_analysis_results(args.output_dir)
            analyzer.print_summary()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Analysis error: {e}")


if __name__ == "__main__":
    main()