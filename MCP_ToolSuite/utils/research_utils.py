# MCP_DeepResearch/utils/research_utils.py

import json
import os
import logging
from typing import Dict, Any, List, Tuple

def parse_research_results(results: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse and validate research results, formatting them for the ReportAgent."""
    contexts = []
    sources = []
    
    for result in results:
        title = result.get("title", "Untitled").strip()
        content = result.get("content", "").strip()
        url = result.get("url", "").strip()
        
        if not content or not url:
            continue

        # Format the context for the LLM to easily understand the source of the info
        formatted_context = f"Source: {title}\nURL: {url}\n\nContent:\n{content}\n\n---\n"
        contexts.append(formatted_context)
        
        sources.append({
            "title": title,
            "url": url,
            "date": result.get("published_date") or "Date not available",
            "type": "article"
        })
    
    return contexts, sources

def format_sources_section(sources: List[Dict[str, str]]) -> str:
    """Format the sources section of the response with proper markdown."""
    if not sources:
        return ""
        
    sources_section = "\n\n## Sources\n"
    for idx, source in enumerate(sources, 1):
        sources_section += f"{idx}. [{source['title']}]({source['url']}) - {source['date']}\n"
        
    return sources_section


def validate_response(response: Any, expected_type: type) -> bool:
    """Validate response type and structure"""
    return isinstance(response, expected_type)

def save_markdown_report(content: str) -> str:
    """Save markdown content to a file"""
    pass

def convert_to_html(markdown_content: str) -> str:
    """Convert markdown to styled HTML"""
    pass
