"""
Image Generation Tool using Hyperbolic API

This script provides an interactive interface for generating images using the Hyperbolic API.
Requires HYPERBOLIC_API_KEY environment variable to be set.
"""

import requests
import json
import base64
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any


class ImageGenerator:
    """Image generator using Hyperbolic API"""
    
    def __init__(self, output_dir: str = "generated_images"):
        self.API_URL = "https://api.hyperbolic.xyz/v1/image/generation"
        self.API_KEY = os.getenv('HYPERBOLIC_API_KEY')
        
        if not self.API_KEY:
            raise ValueError("HYPERBOLIC_API_KEY environment variable is required")
        
        # Create directories if they don't exist
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.history_file = self.output_dir / "generation_history.json"
        
        # Load history if exists
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load generation history from JSON file"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []

    def _save_history(self) -> None:
        """Save generation history to JSON file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def generate_image(self, prompt: str, save: bool = True, 
                      model: str = "FLUX.1-dev", steps: int = 30, 
                      cfg_scale: float = 5, height: int = 1024, 
                      width: int = 1024) -> Optional[str]:
        """Generate image from prompt using Hyperbolic API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        
        data = {
            "model_name": model,
            "prompt": prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "enable_refiner": False,
            "height": height,
            "width": width,
            "backend": "auto"
        }
        
        try:
            print(f"Generating image for prompt: {prompt}")
            response = requests.post(self.API_URL, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            if 'images' in result and result['images']:
                image_data = result['images'][0]['image']
                
                if save:
                    return self.save_generation(prompt, image_data)
                return image_data
            
            raise Exception("No image data in response")
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def save_generation(self, prompt: str, image_data: str) -> str:
        """Save generated image and add to history"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(x for x in prompt[:30] if x.isalnum() or x in (' ', '-', '_'))
        filename = f"{timestamp}_{safe_prompt}"
        
        # Save image
        image_path = self.output_dir / f"{filename}.png"
        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        # Add to history
        entry = {
            "timestamp": timestamp,
            "prompt": prompt,
            "filename": image_path.name
        }
        self.history.append(entry)
        self._save_history()
        
        print(f"Saved image to: {image_path}")
        return image_data

    def list_generations(self) -> None:
        """List all previous generations"""
        if not self.history:
            print("No previous generations found.")
            return
        
        print("\nGeneration History:")
        print("-" * 80)
        for i, entry in enumerate(self.history, 1):
            print(f"{i}. [{entry['timestamp']}] {entry['prompt'][:50]}...")
            print(f"   File: {entry['filename']}")
            print()

    def view_generation(self, index: int) -> None:
        """View details of a specific generation"""
        if 0 <= index < len(self.history):
            entry = self.history[index]
            print("\nGeneration Details:")
            print("-" * 80)
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Prompt: {entry['prompt']}")
            print(f"File: {entry['filename']}")
            
            # Check if file exists
            image_path = self.output_dir / entry['filename']
            if image_path.exists():
                print(f"Image exists at: {image_path}")
            else:
                print("Warning: Image file not found!")
        else:
            print("Invalid generation index!")


def main():
    """Main interactive loop"""
    try:
        generator = ImageGenerator()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set HYPERBOLIC_API_KEY environment variable")
        return
    
    while True:
        print("\nImage Generator Menu:")
        print("1. Generate new image")
        print("2. List previous generations")
        print("3. View generation details")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            prompt = input("\nEnter your prompt: ")
            generator.generate_image(prompt)
            
        elif choice == "2":
            generator.list_generations()
            
        elif choice == "3":
            generator.list_generations()
            try:
                index = int(input("\nEnter the number of the generation to view: ")) - 1
                generator.view_generation(index)
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == "4":
            print("\nGoodbye!")
            break
            
        else:
            print("\nInvalid choice! Please try again.")
        
        time.sleep(1)  # Small pause before showing menu again


if __name__ == "__main__":
    main()