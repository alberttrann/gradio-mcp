import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def validate_response(response: Any, expected_type: type) -> bool:
    """Validate response type and structure"""
    return isinstance(response, expected_type)

def parse_research_results(results: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse and validate research results"""
    contexts = []
    sources = []
    
    for result in results:
        content = result.get('content', '').strip()
        if content:
            contexts.append(content)
            sources.append({
                'title': result.get('title', 'Untitled'),
                'url': result.get('url', ''),
                'date': result.get('published_date', 'Date not available'),
                'type': 'article'
            })
    
    return contexts, sources

def format_sources_section(sources: List[Dict[str, str]]) -> str:
    """Format the sources section with proper markdown"""
    if not sources:
        return "\n\n## Sources\nNo external sources were referenced."
        
    sections = "\n\n## Sources\n\n"
    for idx, source in enumerate(sources, 1):
        sections += f"{idx}. [{source['title']}]({source['url']}) - {source['date']}\n"
            
    return sections

def save_markdown_report(content: str) -> str:
    """Save markdown content to a file"""
    try:
        os.makedirs("generated_reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.md"
        file_path = os.path.join("generated_reports", filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return file_path
    except Exception as e:
        logger.error(f"Failed to save markdown report: {str(e)}")
        raise

def convert_to_html(markdown_content: str) -> str:
    """Convert markdown to styled HTML"""
    try:
        from markdown_it import MarkdownIt
        md = MarkdownIt('commonmark', {'html': True})
        html_content = md.render(markdown_content)
        
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: system-ui, -apple-system, sans-serif;
                    line-height: 1.6;
                    max-width: 900px;
                    margin: 40px auto;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{ color: #2c3e50; }}
                code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
                pre {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        os.makedirs("generated_reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join("generated_reports", f"report_{timestamp}.html")
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(styled_html)
            
        return html_path
        
    except Exception as e:
        logger.error(f"Failed to convert markdown to HTML: {str(e)}")
        raise