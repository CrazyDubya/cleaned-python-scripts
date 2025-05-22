"""
Mandelbrot Set Renderer

Renders the Mandelbrot set in the terminal using ASCII characters.
Automatically adapts to terminal size.
"""

import os
import shutil
import argparse
from typing import Tuple


def mandelbrot(c: complex, max_iter: int) -> int:
    """Calculate the number of iterations for a complex number in the Mandelbrot set"""
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n


def render_mandelbrot(width: int = None, height: int = None, 
                     max_iter: int = 100, 
                     x_range: Tuple[float, float] = (-2.0, 1.0),
                     y_range: Tuple[float, float] = (-1.5, 1.5)) -> None:
    """Render the Mandelbrot set to terminal"""
    
    # Get terminal dimensions if not specified
    if width is None or height is None:
        terminal_width, terminal_height = shutil.get_terminal_size()
        width = width or terminal_width
        height = height or terminal_height

    x_min, x_max = x_range
    y_min, y_max = y_range

    # Calculate step sizes
    x_step = (x_max - x_min) / width
    y_step = (y_max - y_min) / height

    # Characters representing different iteration levels
    chars = " .:-=+*#%@"

    print(f"Rendering Mandelbrot set ({width}x{height}, {max_iter} iterations)")
    print("-" * width)

    # Render the Mandelbrot set
    for y in range(height):
        line = ""
        for x in range(width):
            # Map pixel position to complex plane
            real = x_min + x * x_step
            imag = y_min + y * y_step
            c = complex(real, imag)
            
            # Calculate iterations
            m = mandelbrot(c, max_iter)
            
            # Map iterations to character
            char_index = min(m, len(chars) - 1)
            line += chars[char_index]
            
        print(line)


def render_mandelbrot_zoomed(center: complex, zoom: float, 
                           width: int = None, height: int = None,
                           max_iter: int = 100) -> None:
    """Render a zoomed view of the Mandelbrot set"""
    
    if width is None or height is None:
        terminal_width, terminal_height = shutil.get_terminal_size()
        width = width or terminal_width
        height = height or terminal_height

    # Calculate zoom range
    aspect_ratio = width / height if height > 0 else 1
    zoom_range = 4.0 / zoom
    
    x_range = (center.real - zoom_range * aspect_ratio / 2, 
               center.real + zoom_range * aspect_ratio / 2)
    y_range = (center.imag - zoom_range / 2, 
               center.imag + zoom_range / 2)
    
    print(f"Zoomed view: center={center}, zoom={zoom}x")
    render_mandelbrot(width, height, max_iter, x_range, y_range)


def interactive_explorer() -> None:
    """Interactive Mandelbrot explorer"""
    print("Interactive Mandelbrot Explorer")
    print("Commands:")
    print("  'q' - quit")
    print("  'r' - render at current position")
    print("  'z <zoom>' - set zoom level")
    print("  'i <iterations>' - set max iterations")
    print("  'c <real> <imag>' - set center point")
    print("  'p <x> <y>' - preset interesting locations")
    print("")
    
    # Default parameters
    center = complex(-0.5, 0.0)
    zoom = 1.0
    max_iter = 100
    
    # Interesting presets
    presets = {
        '1': complex(-0.7269, 0.1889),      # Spiral
        '2': complex(-0.8, 0.156),          # Feather
        '3': complex(-0.74529, 0.11307),    # Lightning
        '4': complex(0.285, 0.01),          # Elephant Valley
        '5': complex(-1.25066, 0.02012),    # Seahorse
    }
    
    while True:
        try:
            command = input(f"\nCenter: {center}, Zoom: {zoom}x, Iterations: {max_iter}\n> ").strip()
            
            if not command:
                continue
                
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'q':
                break
            elif cmd == 'r':
                render_mandelbrot_zoomed(center, zoom, max_iter=max_iter)
            elif cmd == 'z' and len(parts) > 1:
                zoom = float(parts[1])
                print(f"Zoom set to {zoom}x")
            elif cmd == 'i' and len(parts) > 1:
                max_iter = int(parts[1])
                print(f"Max iterations set to {max_iter}")
            elif cmd == 'c' and len(parts) > 2:
                center = complex(float(parts[1]), float(parts[2]))
                print(f"Center set to {center}")
            elif cmd == 'p' and len(parts) > 1:
                preset = parts[1]
                if preset in presets:
                    center = presets[preset]
                    zoom = 100.0  # Good zoom for presets
                    print(f"Moved to preset {preset}: {center}")
                else:
                    print(f"Available presets: {', '.join(presets.keys())}")
            else:
                print("Invalid command. Type 'q' to quit.")
                
        except (ValueError, IndexError):
            print("Invalid input format.")
        except KeyboardInterrupt:
            break
    
    print("Goodbye!")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Render the Mandelbrot set")
    parser.add_argument("--width", type=int, help="Output width in characters")
    parser.add_argument("--height", type=int, help="Output height in characters")
    parser.add_argument("--iterations", type=int, default=100, 
                       help="Maximum iterations (default: 100)")
    parser.add_argument("--center-real", type=float, default=-0.5,
                       help="Real part of center point (default: -0.5)")
    parser.add_argument("--center-imag", type=float, default=0.0,
                       help="Imaginary part of center point (default: 0.0)")
    parser.add_argument("--zoom", type=float, default=1.0,
                       help="Zoom level (default: 1.0)")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive explorer")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_explorer()
    else:
        center = complex(args.center_real, args.center_imag)
        if args.zoom != 1.0:
            render_mandelbrot_zoomed(center, args.zoom, args.width, args.height, args.iterations)
        else:
            render_mandelbrot(args.width, args.height, args.iterations)


if __name__ == "__main__":
    main()