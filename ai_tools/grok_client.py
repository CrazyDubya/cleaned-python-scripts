"""
Grok AI Client

A simple client for interacting with Grok (X.AI) API.
Requires XAI_API_KEY environment variable to be set.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from tenacity import retry, wait_exponential


class GrokClient:
    """Client for Grok AI API"""
    
    def __init__(self):
        self.API_URL = "https://api.x.ai/v1/chat/completions"
        self.DEFERRED_URL = "https://api.x.ai/v1/chat/deferred-completion"
        self.API_KEY = os.getenv('XAI_API_KEY')
        
        if not self.API_KEY:
            raise ValueError("XAI_API_KEY environment variable is required")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }

    def start_chat(self, messages: List[Dict[str, str]], 
                   model: str = "grok-3-beta", 
                   system_prompt: Optional[str] = None) -> str:
        """Start a deferred chat completion and return request ID"""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "messages": full_messages,
            "model": model,
            "deferred": True
        }

        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["request_id"]

    @retry(wait=wait_exponential(multiplier=1, min=1, max=60))
    def get_completion(self, request_id: str) -> Dict[str, Any]:
        """Get completion result by request ID (with retry for deferred responses)"""
        response = requests.get(f"{self.DEFERRED_URL}/{request_id}", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            raise Exception("Response not ready yet")
        else:
            response.raise_for_status()

    def run_prompt(self, user_prompt: str, 
                   system_prompt: str = "You are a creative and observant assistant.", 
                   model: str = "grok-3-beta") -> str:
        """Run a simple prompt and return the response"""
        request_id = self.start_chat(
            messages=[{"role": "user", "content": user_prompt}],
            model=model,
            system_prompt=system_prompt
        )
        print(f"Request ID: {request_id}")
        completion = self.get_completion(request_id)
        return completion['choices'][0]['message']['content']

    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = "grok-3-beta",
                       system_prompt: Optional[str] = None) -> str:
        """Complete a chat conversation"""
        request_id = self.start_chat(messages, model, system_prompt)
        completion = self.get_completion(request_id)
        return completion['choices'][0]['message']['content']


def main():
    """Example usage"""
    try:
        client = GrokClient()
        
        # Simple prompt example
        prompt = "Explain quantum computing in simple terms"
        response = client.run_prompt(prompt)
        print(f"Response: {response}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set XAI_API_KEY environment variable")
    except Exception as e:
        print(f"API Error: {e}")


if __name__ == "__main__":
    main()