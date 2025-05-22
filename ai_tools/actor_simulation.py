"""
Actor Simulation using Grok AI

This script simulates AI actors with different personalities and behaviors
using the Grok AI API for generating responses.
"""

import os
import json
import uuid
import random
import re
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional

from grok_client import GrokClient


class Actor:
    """AI Actor with personality and memory"""
    
    def __init__(self, name: str, archetype: str, agency: int, memory: Optional[List[str]] = None):
        self.name = name
        self.archetype = archetype
        self.agency = agency
        self.meta_memory = memory or []
        self.emotional_state = "neutral"
        self.internal_thoughts = []
        self.grok_client = GrokClient()

    def think(self, cue_text: str = "") -> str:
        """Generate thoughts and actions based on current state and cues"""
        prompt = f"""
You are {self.name}, a {self.archetype} actor inside an experimental simulation.
Your current emotional state is '{self.emotional_state}'.
Recent memory includes: {", ".join(self.meta_memory[-3:]) or 'Nothing'}.
Active cue: {cue_text or 'none'}.

Respond in this format:
THOUGHT: (your internal reflection)
PERCEPTION: (what you notice)
ACTION: (what you attempt or prepare)
EMOTION: (how your emotion shifts or deepens)
"""
        try:
            response = self.grok_client.run_prompt(
                user_prompt=prompt,
                system_prompt=f"You are {self.name}, embodying the {self.archetype} archetype."
            )
            self.internal_thoughts.append(response)
            
            # Update emotional state if mentioned in response
            emotion_match = re.search(r'EMOTION:\s*(.+)', response)
            if emotion_match:
                self.emotional_state = emotion_match.group(1).strip()
                
            # Add to memory
            if len(self.meta_memory) > 10:  # Keep memory manageable
                self.meta_memory.pop(0)
            self.meta_memory.append(f"Responded to: {cue_text[:50]}...")
            
            return response
            
        except Exception as e:
            error_response = f"[ERROR]: {e}"
            self.internal_thoughts.append(error_response)
            return error_response

    def get_state(self) -> Dict[str, Any]:
        """Get current actor state"""
        return {
            "name": self.name,
            "archetype": self.archetype,
            "agency": self.agency,
            "emotional_state": self.emotional_state,
            "memory_items": len(self.meta_memory),
            "thoughts_count": len(self.internal_thoughts)
        }


class VoidSimulation:
    """Simulation environment for AI actors"""
    
    def __init__(self):
        self.actors: List[Actor] = []
        self.tick_count = 0
        self.simulation_log = []

    def add_actor(self, name: str, archetype: str, agency: int) -> Actor:
        """Add a new actor to the simulation"""
        actor = Actor(name, archetype, agency)
        self.actors.append(actor)
        return actor

    def simulate_tick(self, cue: Optional[str] = None) -> Dict[str, Any]:
        """Run one simulation tick"""
        self.tick_count += 1
        tick_results = {
            "tick": self.tick_count,
            "cue": cue,
            "timestamp": datetime.now().isoformat(),
            "actor_responses": []
        }
        
        print(f"\n===== VOID TICK {self.tick_count} BEGIN =====")
        if cue:
            print(f"Active Cue: {cue}")
        
        for actor in self.actors:
            print(f"\n[{actor.name}]")
            response = actor.think(cue or "")
            print(response)
            
            tick_results["actor_responses"].append({
                "actor": actor.name,
                "response": response,
                "state": actor.get_state()
            })
        
        print("===== VOID TICK END =====\n")
        
        self.simulation_log.append(tick_results)
        return tick_results

    def run_simulation(self, cues: List[str], ticks_per_cue: int = 1) -> None:
        """Run a full simulation with multiple cues"""
        for cue in cues:
            for _ in range(ticks_per_cue):
                self.simulate_tick(cue)

    def save_log(self, filename: str = None) -> None:
        """Save simulation log to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_log_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.simulation_log, f, indent=2)
        print(f"Simulation log saved to {filename}")

    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary"""
        return {
            "total_ticks": self.tick_count,
            "actors": [actor.get_state() for actor in self.actors],
            "log_entries": len(self.simulation_log)
        }


def create_default_actors() -> List[Dict[str, Any]]:
    """Create default actor configurations"""
    return [
        {"name": "Orra", "archetype": "Archivist", "agency": 42},
        {"name": "Cellen", "archetype": "Disbeliever", "agency": 87},
        {"name": "Jun", "archetype": "Trickster", "agency": 69},
    ]


def main():
    """Main simulation runner"""
    try:
        simulation = VoidSimulation()
        
        # Add default actors
        actors_config = create_default_actors()
        for config in actors_config:
            simulation.add_actor(**config)
        
        # Define simulation cues
        cues = [
            "A low-frequency hum begins to pulse beneath the floorboards.",
            "The lights flicker momentarily, then return to normal.",
            "A door that was previously locked now stands slightly ajar.",
            "The temperature in the room drops noticeably.",
            "An unfamiliar shadow moves across the wall."
        ]
        
        # Run simulation
        print("Starting AI Actor Simulation...")
        simulation.run_simulation(cues, ticks_per_cue=1)
        
        # Save results
        simulation.save_log()
        
        # Print summary
        summary = simulation.get_summary()
        print(f"\nSimulation Complete:")
        print(f"Total Ticks: {summary['total_ticks']}")
        print(f"Actors: {len(summary['actors'])}")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please set XAI_API_KEY environment variable")
    except Exception as e:
        print(f"Simulation Error: {e}")


if __name__ == "__main__":
    main()