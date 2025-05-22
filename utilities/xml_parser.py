"""
Robust XML Parser

Advanced XML parsing utility with multiple fallback strategies and error handling.
Handles malformed XML, encoding issues, and provides salvage mechanisms.
"""

import xml.etree.ElementTree as ET
from io import StringIO
import re
from html import unescape
from typing import Optional, List, Tuple


class RobustXMLParser:
    """Robust XML parser with multiple parsing strategies"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.parse_attempts = []
        
    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)
            
    def _remove_invalid_chars(self, text: str) -> str:
        """Remove invalid XML characters"""
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    def _fix_common_xml_issues(self, text: str) -> str:
        """Attempt to fix common XML issues"""
        # Replace non-XML ampersands
        text = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', text)
        
        # Attempt to close unclosed tags
        open_tags = []
        for match in re.finditer(r'<(\w+)[^>]*>', text):
            tag = match.group(1)
            if f'</{tag}>' not in text[match.end():]:
                open_tags.append(tag)
                
        # Close unclosed tags in reverse order
        for tag in reversed(open_tags):
            text += f'</{tag}>'
            
        return text
    
    def _parse_with_strategy(self, xml_string: str, strategy_name: str, 
                           parse_func) -> Optional[ET.Element]:
        """Attempt parsing with a specific strategy"""
        try:
            self._log(f"Trying strategy: {strategy_name}")
            result = parse_func()
            self.parse_attempts.append((strategy_name, "Success"))
            self._log(f"Success with strategy: {strategy_name}")
            return result
        except ET.ParseError as e:
            self.parse_attempts.append((strategy_name, f"ParseError: {e}"))
            self._log(f"ParseError with {strategy_name}: {e}")
        except Exception as e:
            self.parse_attempts.append((strategy_name, f"Error: {e}"))
            self._log(f"Error with {strategy_name}: {e}")
        return None
    
    def parse_xml(self, xml_string: str) -> Optional[ET.Element]:
        """
        Attempt to parse XML with multiple strategies and advanced error handling.
        
        Args:
            xml_string: The XML content to parse
            
        Returns:
            Parsed XML element or None if all strategies fail
        """
        if not xml_string:
            self._log("Empty XML string provided.")
            return None

        # Reset parse attempts for this parsing session
        self.parse_attempts = []
        
        # Clean and prepare the XML string
        xml_string = unescape(xml_string.strip())

        # Strategy 1: Direct parsing with invalid character removal
        result = self._parse_with_strategy(
            xml_string,
            "Direct parsing (cleaned)",
            lambda: ET.fromstring(self._remove_invalid_chars(xml_string))
        )
        if result is not None:
            return result

        # Strategy 2: Wrap in root element and parse
        result = self._parse_with_strategy(
            xml_string,
            "Root-wrapped parsing",
            lambda: ET.parse(StringIO(
                self._remove_invalid_chars(f"<root>{xml_string}</root>")
            )).getroot()
        )
        if result is not None:
            return result

        # Strategy 3: Fix common issues then parse
        fixed_xml = self._fix_common_xml_issues(xml_string)
        result = self._parse_with_strategy(
            fixed_xml,
            "Fixed issues parsing",
            lambda: ET.fromstring(self._remove_invalid_chars(fixed_xml))
        )
        if result is not None:
            return result

        # Strategy 4: Fix issues + wrap in root
        result = self._parse_with_strategy(
            fixed_xml,
            "Fixed + wrapped parsing",
            lambda: ET.parse(StringIO(
                self._remove_invalid_chars(f"<root>{fixed_xml}</root>")
            )).getroot()
        )
        if result is not None:
            return result

        # Strategy 5: Extract valid XML declaration + content
        self._log("Attempting to extract valid XML substrings...")
        xml_declaration_pattern = r'<\?xml.*?>.*?</.*?>'
        matches = re.findall(xml_declaration_pattern, xml_string, re.DOTALL | re.IGNORECASE)
        
        for i, match in enumerate(matches):
            result = self._parse_with_strategy(
                match,
                f"XML declaration extraction #{i+1}",
                lambda m=match: ET.fromstring(
                    self._remove_invalid_chars(self._fix_common_xml_issues(m))
                )
            )
            if result is not None:
                return result

        # Strategy 6: Salvage XML-like structure
        self._log("Attempting to salvage XML-like structure...")
        element_pattern = r'<(\w+).*?>(.*?)</\1>'
        matches = re.findall(element_pattern, xml_string, re.DOTALL | re.IGNORECASE)
        
        if matches:
            try:
                root = ET.Element('root')
                for tag, content in matches:
                    elem = ET.SubElement(root, tag)
                    elem.text = content.strip()
                self.parse_attempts.append(("XML structure salvage", "Success"))
                self._log("Successfully salvaged XML structure")
                return root
            except Exception as e:
                self.parse_attempts.append(("XML structure salvage", f"Error: {e}"))
                self._log(f"Failed to salvage XML structure: {e}")

        # Strategy 7: Create minimal structure from any tag-like content
        self._log("Creating minimal structure from tag-like content...")
        simple_tag_pattern = r'<(\w+)[^>]*>([^<]*)'
        simple_matches = re.findall(simple_tag_pattern, xml_string)
        
        if simple_matches:
            try:
                root = ET.Element('document')
                for tag, content in simple_matches[:10]:  # Limit to first 10 matches
                    elem = ET.SubElement(root, tag)
                    elem.text = content.strip() if content.strip() else None
                self.parse_attempts.append(("Minimal structure creation", "Success"))
                self._log("Successfully created minimal structure")
                return root
            except Exception as e:
                self.parse_attempts.append(("Minimal structure creation", f"Error: {e}"))
                self._log(f"Failed to create minimal structure: {e}")

        self._log("All parsing strategies failed.")
        return None
    
    def get_parse_report(self) -> List[Tuple[str, str]]:
        """Get a report of all parsing attempts made"""
        return self.parse_attempts.copy()
    
    def print_parse_report(self) -> None:
        """Print a formatted report of parsing attempts"""
        print("\nXML Parsing Report:")
        print("-" * 50)
        for strategy, result in self.parse_attempts:
            print(f"{strategy}: {result}")


def xml_to_dict(element: ET.Element) -> dict:
    """Convert XML element to dictionary representation"""
    result = {}
    
    # Add attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Add text content
    if element.text and element.text.strip():
        if len(element) == 0:  # No children, just text
            return element.text.strip()
        else:
            result['@text'] = element.text.strip()
    
    # Add children
    for child in element:
        child_data = xml_to_dict(child)
        
        if child.tag in result:
            # Handle multiple children with same tag
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    
    return result


def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust XML Parser")
    parser.add_argument("file", nargs='?', help="XML file to parse")
    parser.add_argument("--text", help="XML text to parse directly")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--to-dict", action="store_true", help="Convert to dictionary")
    
    args = parser.parse_args()
    
    xml_parser = RobustXMLParser(verbose=args.verbose)
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                xml_content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.text:
        xml_content = args.text
    else:
        # Example XML for testing
        xml_content = '''<?xml version="1.0"?>
        <root>
            <item id="1">First item</item>
            <item id="2">Second item & more</item>
            <unclosed_tag>This tag is not closed
        </root>'''
        print("Using example XML (use --file or --text for your own content)")
    
    print(f"Parsing XML content ({len(xml_content)} characters)...")
    
    result = xml_parser.parse_xml(xml_content)
    
    if result is not None:
        print("\n✓ Successfully parsed XML!")
        print(f"Root element: {result.tag}")
        print(f"Children: {len(list(result))}")
        
        if args.to_dict:
            print("\nDictionary representation:")
            import json
            dict_result = xml_to_dict(result)
            print(json.dumps(dict_result, indent=2, ensure_ascii=False))
        else:
            print("\nFirst few children:")
            for i, child in enumerate(list(result)[:5]):
                print(f"  {child.tag}: {child.text[:50] if child.text else 'No text'}...")
                
    else:
        print("\n✗ Failed to parse XML with all strategies.")
    
    if args.verbose:
        xml_parser.print_parse_report()


if __name__ == "__main__":
    main()