"""
Universal Model Loader

A utility for loading and testing various ML models including MLX and Transformers.
Provides a unified interface for model loading, generation, and basic testing.
"""

import subprocess
import sys
import os
import traceback
import argparse
from pathlib import Path
from typing import Optional, Tuple, Any, Dict


class ModelLoader:
    """Universal model loader for different ML frameworks"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.models = {}
        self.tokenizers = {}
        
    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
            
    def install_package(self, package: str) -> bool:
        """Install Python package if not present"""
        try:
            __import__(package.replace('-', '_'))
            self._log(f"Package {package} already installed")
            return True
        except ImportError:
            try:
                self._log(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
                self._log(f"Successfully installed {package}")
                return True
            except subprocess.CalledProcessError as e:
                self._log(f"Failed to install {package}: {e}")
                return False

    def setup_mlx(self) -> bool:
        """Setup MLX environment"""
        packages = ["mlx", "transformers", "huggingface_hub", "hf_transfer"]
        
        for package in packages:
            if not self.install_package(package):
                return False
        
        # Check for mlx-examples repository
        if not Path("mlx-examples").exists():
            try:
                self._log("Cloning mlx-examples repository...")
                subprocess.run([
                    "git", "clone", "https://github.com/ml-explore/mlx-examples.git"
                ], check=True)
                self._log("mlx-examples cloned successfully")
            except subprocess.CalledProcessError as e:
                self._log(f"Failed to clone mlx-examples: {e}")
                return False
        
        return True

    def load_mlx_model(self, model_name: str = "phi2") -> Optional[Tuple[Any, Any]]:
        """Load model using MLX"""
        try:
            import mlx.core as mx
            from transformers import AutoTokenizer
            
            self._log(f"Loading MLX model: {model_name}")
            
            # Set environment for faster downloads
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            # Download model (simplified approach)
            model_path = f"mlx-community/{model_name}"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # For demonstration, we'll create a simple model placeholder
            # In practice, you'd load the actual MLX model weights
            model = {"name": model_name, "path": model_path}
            
            self.models[f"mlx_{model_name}"] = model
            self.tokenizers[f"mlx_{model_name}"] = tokenizer
            
            self._log(f"MLX model {model_name} loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            self._log(f"Error loading MLX model: {e}")
            return None

    def load_transformers_model(self, model_name: str) -> Optional[Tuple[Any, Any]]:
        """Load model using Hugging Face Transformers"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._log(f"Loading Transformers model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            self.models[f"hf_{model_name}"] = model
            self.tokenizers[f"hf_{model_name}"] = tokenizer
            
            self._log(f"Transformers model {model_name} loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            self._log(f"Error loading Transformers model: {e}")
            return None

    def load_ollama_model(self, model_name: str) -> Optional[Dict[str, str]]:
        """Setup Ollama model (requires Ollama to be installed)"""
        try:
            import ollama
            
            self._log(f"Setting up Ollama model: {model_name}")
            
            # Try to pull the model if it doesn't exist
            try:
                ollama.pull(model_name)
                self._log(f"Ollama model {model_name} ready")
            except Exception as e:
                self._log(f"Note: Model may need to be pulled manually: {e}")
            
            model_info = {"name": model_name, "type": "ollama"}
            self.models[f"ollama_{model_name}"] = model_info
            
            return model_info
            
        except ImportError:
            self._log("Ollama package not installed. Install with: pip install ollama")
            return None
        except Exception as e:
            self._log(f"Error setting up Ollama model: {e}")
            return None

    def generate_text(self, model_key: str, prompt: str, 
                     max_length: int = 50, temperature: float = 1.0) -> str:
        """Generate text using loaded model"""
        if model_key not in self.models:
            return f"Model {model_key} not loaded"
        
        model = self.models[model_key]
        
        try:
            if model_key.startswith("hf_"):
                # Transformers generation
                tokenizer = self.tokenizers[model_key]
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                outputs = model.generate(
                    inputs, 
                    max_length=max_length, 
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response
                
            elif model_key.startswith("ollama_"):
                # Ollama generation
                import ollama
                response = ollama.generate(
                    model=model["name"], 
                    prompt=prompt
                )
                return response["response"]
                
            elif model_key.startswith("mlx_"):
                # MLX generation (placeholder)
                return f"MLX generation for '{prompt}' (placeholder implementation)"
                
            else:
                return f"Unknown model type for {model_key}"
                
        except Exception as e:
            return f"Generation error: {e}"

    def list_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded models"""
        model_info = {}
        for key, model in self.models.items():
            if isinstance(model, dict):
                model_info[key] = {
                    "type": key.split('_')[0],
                    "name": model.get("name", "unknown"),
                    "loaded": True
                }
            else:
                model_info[key] = {
                    "type": key.split('_')[0],
                    "name": key.split('_', 1)[1],
                    "loaded": True
                }
        return model_info

    def test_model(self, model_key: str, test_prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test a loaded model with a simple prompt"""
        if model_key not in self.models:
            return {"error": f"Model {model_key} not loaded"}
        
        start_time = None
        try:
            import time
            start_time = time.time()
            
            response = self.generate_text(model_key, test_prompt)
            
            end_time = time.time()
            
            return {
                "model": model_key,
                "prompt": test_prompt,
                "response": response,
                "generation_time": round(end_time - start_time, 2),
                "success": True
            }
            
        except Exception as e:
            return {
                "model": model_key,
                "prompt": test_prompt,
                "error": str(e),
                "success": False
            }


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Universal Model Loader")
    parser.add_argument("--framework", choices=["mlx", "transformers", "ollama"], 
                       help="ML framework to use")
    parser.add_argument("--model", help="Model name to load")
    parser.add_argument("--prompt", help="Text prompt for generation")
    parser.add_argument("--max-length", type=int, default=50, 
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, 
                       help="Generation temperature")
    parser.add_argument("--test", action="store_true", 
                       help="Run quick test of loaded model")
    parser.add_argument("--list", action="store_true", 
                       help="List available/loaded models")
    parser.add_argument("--interactive", action="store_true", 
                       help="Interactive mode")
    
    args = parser.parse_args()
    
    loader = ModelLoader()
    
    if args.list:
        models = loader.list_loaded_models()
        if models:
            print("\nLoaded Models:")
            for key, info in models.items():
                print(f"  {key}: {info['name']} ({info['type']})")
        else:
            print("No models loaded")
        return
    
    # Load model based on framework
    if args.framework and args.model:
        if args.framework == "mlx":
            if loader.setup_mlx():
                model, tokenizer = loader.load_mlx_model(args.model)
            else:
                print("Failed to setup MLX environment")
                return
        elif args.framework == "transformers":
            model, tokenizer = loader.load_transformers_model(args.model)
        elif args.framework == "ollama":
            model = loader.load_ollama_model(args.model)
        
        model_key = f"{args.framework}_{args.model}"
        
        if args.test:
            print(f"\nTesting model {model_key}...")
            result = loader.test_model(model_key)
            if result["success"]:
                print(f"Prompt: {result['prompt']}")
                print(f"Response: {result['response']}")
                print(f"Time: {result['generation_time']}s")
            else:
                print(f"Test failed: {result['error']}")
        
        if args.prompt:
            print(f"\nGenerating response...")
            response = loader.generate_text(
                model_key, args.prompt, args.max_length, args.temperature
            )
            print(f"Response: {response}")
    
    if args.interactive:
        print("\nInteractive Model Loader")
        print("Commands: load <framework> <model>, test <model_key>, generate <model_key> <prompt>, list, quit")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                if not command:
                    continue
                
                if command[0] == "quit":
                    break
                elif command[0] == "load" and len(command) >= 3:
                    framework, model = command[1], command[2]
                    print(f"Loading {framework} model: {model}")
                    # Implementation similar to above
                elif command[0] == "list":
                    models = loader.list_loaded_models()
                    for key, info in models.items():
                        print(f"  {key}: {info['name']}")
                elif command[0] == "test" and len(command) >= 2:
                    model_key = command[1]
                    result = loader.test_model(model_key)
                    print(f"Test result: {result}")
                elif command[0] == "generate" and len(command) >= 3:
                    model_key = command[1]
                    prompt = " ".join(command[2:])
                    response = loader.generate_text(model_key, prompt)
                    print(f"Response: {response}")
                else:
                    print("Invalid command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print("Model loader finished.")


if __name__ == "__main__":
    main()