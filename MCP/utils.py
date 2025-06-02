import json
from typing import Dict, Any, Optional, List, Tuple

def validate_response(response: Any, expected_type: type) -> bool:
    """Validate response type and structure"""
    if not isinstance(response, expected_type):
        return False
    return True

def format_source_content(
    title: str,
    url: str,
    date: str,
    content: str,
    source_type: str
) -> str:
    """Format source content with consistent styling"""
    return f"""### Source: {title}
URL: {url}
Date: {date if date else 'Not available'}
Type: {source_type}

**Key Content:**
{content}

---"""

def parse_research_results(results: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse and validate research results"""
    contexts = []
    sources = []
    
    for result in results:
        title = result.get("title", "").strip()
        content = result.get("content", "").strip()
        url = result.get("url", "").strip()
        date = result.get("published_date", "").strip()
        
        if title and content:
            source_type = (
                "research_paper"
                if "arxiv.org" in url or "paper" in url.lower()
                else "article"
            )
            
            sources.append({
                "title": title,
                "url": url,
                "date": date if date else "Date not available",
                "type": source_type
            })
            
            contexts.append(
                format_source_content(title, url, date, content, source_type)
            )
    
    return contexts, sources

def format_sources_section(sources: List[Dict[str, str]]) -> str:
    """Format the sources section of the response"""
    sources_section = "\n\n## Sources Cited\n\n"
    
    if not sources:
        return sources_section + "No sources were found during the research phase."
        
    research_papers = [s for s in sources if s['type'] == 'research_paper']
    articles = [s for s in sources if s['type'] == 'article']
    
    if research_papers:
        sources_section += "\n### Research Papers\n"
        for idx, source in enumerate(research_papers, 1):
            sources_section += f"{idx}. [{source['title']}]({source['url']}) - {source['date']}\n"
    
    if articles:
        sources_section += "\n### Technical Articles & Resources\n"
        for idx, source in enumerate(articles, 1):
            sources_section += f"{idx}. [{source['title']}]({source['url']}) - {source['date']}\n"
            
    return sources_section
