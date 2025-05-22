"""
Text Refinement Tool using Ollama

This script uses Ollama to iteratively refine and improve text content
through multiple passes with different refinement strategies.
"""

import re
import json
import string
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    import ollama
except ImportError:
    print("Warning: ollama package not installed. Install with: pip install ollama")
    ollama = None


class TextRefiner:
    """Text refinement tool using Ollama LLM"""
    
    def __init__(self, model: str = "gemma2:2b", system_prompt: str = None):
        if not ollama:
            raise ImportError("ollama package is required. Install with: pip install ollama")
            
        self.model = model
        self.system_prompt = system_prompt or "You are helping to refine a list of items."
        self.stats = {
            'turn_items': 0,
            'turn_characters': 0,
            'total_items': 0,
            'total_characters': 0
        }
        
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input by removing excessive symbols and whitespace"""
        # Remove excessive whitespace, punctuation, and control characters
        sanitized_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
        return sanitized_text

    def generate_response(self, prompt: str, context: List[Dict[str, str]] = None) -> str:
        """Generate LLM response using Ollama"""
        try:
            messages = [{'role': 'system', 'content': self.system_prompt}]
            
            if context:
                messages.extend(context)
                
            messages.append({'role': 'user', 'content': prompt})
            
            response = ollama.chat(
                model=self.model,
                messages=messages
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def parse_response(self, response: str) -> List[str]:
        """Parse LLM response using multiple strategies"""
        
        # Strategy 1: Try parsing as JSON
        try:
            result = json.loads(response)
            if isinstance(result, list):
                print("Successfully parsed as JSON list.")
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: Split by commas
        if ',' in response:
            result = [item.strip() for item in response.split(',') if item.strip()]
            if result:
                print("Successfully parsed as comma-separated list.")
                return result

        # Strategy 3: Split by newlines
        if '\n' in response:
            result = [item.strip() for item in response.splitlines() if item.strip()]
            if result:
                print("Successfully parsed as newline-separated list.")
                return result

        # Strategy 4: Extract numbered list items
        numbered_items = re.findall(r'\d+\.\s*(.*)', response)
        if numbered_items:
            print("Successfully parsed using numbered list regex.")
            return numbered_items

        # Strategy 5: Extract bullet points
        bullet_items = re.findall(r'[-*]\s*(.*)', response)
        if bullet_items:
            print("Successfully parsed using bullet point regex.")
            return bullet_items

        # Strategy 6: Split by semicolons
        if ';' in response:
            result = [item.strip() for item in response.split(';') if item.strip()]
            if result:
                print("Successfully parsed as semicolon-separated list.")
                return result

        # Fallback: return as single item
        print("All parsing attempts failed. Returning raw response.")
        return [response.strip()] if response.strip() else []

    def refine_text_portion(self, text_list: List[str], topic: str, 
                          start_idx: int, end_idx: int, 
                          context: List[Dict[str, str]] = None) -> tuple:
        """Refine a portion of the text list"""
        portion = text_list[start_idx:end_idx]
        
        prompt = f"""Refine the following items on the topic '{topic}': {portion}
        
Please improve clarity, coherence, and relevance while maintaining the original meaning.
Return the refined items as a clear list."""

        response = self.generate_response(prompt, context)
        refined_items = self.parse_response(response)

        # Update statistics
        self.stats['turn_items'] = len(refined_items)
        self.stats['turn_characters'] = sum(len(item) for item in refined_items)
        
        # Clean items
        refined_items = [item.strip() for item in refined_items if item.strip()]

        # Fallback to original if refinement failed
        if not refined_items:
            print("Refinement failed, returning original portion.")
            return portion, context

        # Update context for next iteration
        new_context = context or []
        if len(new_context) > 5:  # Keep context manageable
            new_context = new_context[-3:]
        new_context.append({
            'role': 'assistant', 
            'content': response
        })

        return refined_items, new_context

    def recursive_refinement(self, initial_list: List[str], topic: str, 
                           refinement_stages: List[tuple], 
                           context: List[Dict[str, str]] = None) -> List[str]:
        """Apply recursive refinement with multiple stages"""
        master_list = initial_list.copy()
        current_context = context or []

        for stage, (percentage, passes) in enumerate(refinement_stages):
            print(f"\nStage {stage + 1}: {percentage}% portions, {passes} passes")
            
            for pass_num in range(passes):
                portion_size = max(1, len(master_list) * percentage // 100)
                start_idx = (pass_num * portion_size) % len(master_list)
                end_idx = min(start_idx + portion_size, len(master_list))

                print(f"  Pass {pass_num + 1}: refining items {start_idx}-{end_idx}")
                
                refined_portion, current_context = self.refine_text_portion(
                    master_list, topic, start_idx, end_idx, current_context
                )
                
                master_list[start_idx:end_idx] = refined_portion

                # Update statistics
                self.stats['total_characters'] += self.stats['turn_characters']
                self.stats['total_items'] += self.stats['turn_items']

        return master_list

    def save_results(self, results: List[str], filename: str = None) -> str:
        """Save refinement results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"refined_text_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== Text Refinement Results ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Total items: {len(results)}\n\n")
            
            for i, item in enumerate(results, 1):
                f.write(f"{i}. {item}\n")

        return filename

    def get_statistics(self) -> Dict[str, Any]:
        """Get refinement statistics"""
        return self.stats.copy()


def create_default_refinement_stages() -> List[tuple]:
    """Create default refinement stages"""
    return [
        (50, 2),   # Refine 50% of items in 2 passes
        (25, 4),   # Refine 25% in 4 passes  
        (10, 10),  # Refine 10% in 10 passes
        (5, 20)    # Refine 5% in 20 passes (fine-tuning)
    ]


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Refine text using Ollama")
    parser.add_argument("--topic", required=True, help="Topic for refinement")
    parser.add_argument("--text", help="Text to refine directly")
    parser.add_argument("--file", help="File containing text to refine")
    parser.add_argument("--model", default="gemma2:2b", help="Ollama model to use")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--stages", help="Custom refinement stages as JSON")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    try:
        refiner = TextRefiner(model=args.model)
    except ImportError as e:
        print(f"Error: {e}")
        return

    # Get text input
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.text:
        raw_text = args.text
    elif args.interactive:
        print("Enter your text to refine (press Ctrl+D when done):")
        import sys
        raw_text = sys.stdin.read()
    else:
        print("Error: Please provide text via --text, --file, or --interactive")
        return

    # Sanitize input
    sanitized_text = refiner.sanitize_input(raw_text)
    initial_list = [sanitized_text]  # Start with text as single item

    # Parse refinement stages
    if args.stages:
        try:
            stages = json.loads(args.stages)
        except Exception as e:
            print(f"Error parsing custom stages: {e}")
            stages = create_default_refinement_stages()
    else:
        stages = create_default_refinement_stages()

    print(f"Starting refinement with {len(stages)} stages...")
    print(f"Topic: {args.topic}")
    print(f"Model: {args.model}")
    print(f"Initial text length: {len(raw_text)} characters")

    # Perform refinement
    try:
        refined_results = refiner.recursive_refinement(
            initial_list, args.topic, stages
        )

        # Save results
        output_file = refiner.save_results(refined_results, args.output)
        
        # Show summary
        stats = refiner.get_statistics()
        print(f"\nRefinement complete!")
        print(f"Results saved to: {output_file}")
        print(f"Final items: {len(refined_results)}")
        print(f"Total processing: {stats['total_items']} items, {stats['total_characters']} characters")

        # Show first few results
        print(f"\nFirst few refined items:")
        for i, item in enumerate(refined_results[:3], 1):
            print(f"{i}. {item[:100]}{'...' if len(item) > 100 else ''}")

    except Exception as e:
        print(f"Refinement error: {e}")


if __name__ == "__main__":
    main()