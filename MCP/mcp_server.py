# app.py (HTTP SSE Gradio MCP Server - same as before)

import os
import gradio as gr
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:7860")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "My RAG MCP Server")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env")

# --- Initialize API Clients ---
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def research_with_tavily(query: str, max_results: int = 12) -> tuple[str, list]:
    try:
        print(f"Tavily: Performing comprehensive research for '{query}'")
        
        # More comprehensive query expansion
        expanded_query = f"""Comprehensive technical analysis and state-of-the-art review of: {query}
        Must include:
        - Theoretical foundations and mathematical principles
        - Core algorithms and architectural details
        - Implementation strategies and best practices
        - Performance benchmarks and comparisons
        - Latest research papers and findings
        - Real-world applications and case studies
        - Current challenges and limitations
        - Future research directions"""
        
        response = tavily_client.search(
            query=expanded_query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=[
                "arxiv.org", "papers.ssrn.com", "github.com", 
                "towardsdatascience.com", "medium.com", "distill.pub",
                "openai.com", "research.google", "microsoft.com/research",
                "paperswithcode.com", "huggingface.co/blog",
                "neurips.cc", "icml.cc", "iclr.cc", "jmlr.org",
                "science.org", "nature.com", "acm.org", "ieee.org"
            ],
            include_answer=True,
            search_type="comprehensive",
            search_params={
                "include_images": True,
                "include_code": True,
                "time_window": "1y"  # Focus on recent content
            }
        )
        
        if response and response.get("results"):
            contexts = []
            sources = []
            
            for result in response["results"]:
                title = result.get("title", "").strip()
                content = result.get("content", "").strip()
                url = result.get("url", "").strip()
                date = result.get("published_date", "").strip()
                
                if title and content:
                    # Enhanced metadata tracking
                    sources.append({
                        "title": title,
                        "url": url,
                        "date": date if date else "Date not available",
                        "type": "research_paper" if "arxiv.org" in url or "paper" in url.lower() else "article"
                    })
                    
                    # Improved content formatting
                    source_block = f"""### Source: {title}
URL: {url}
Date: {date if date else 'Not available'}
Type: {sources[-1]['type']}

**Key Content:**
{content}

---"""
                    contexts.append(source_block)
            
            context = "\n\n".join(contexts)
            print(f"Tavily: Found {len(contexts)} high-quality sources ({len([s for s in sources if s['type'] == 'research_paper'])} research papers)")
            return context, sources
        else:
            return "No relevant information found by Tavily", []
            
    except Exception as e:
        print(f"Tavily error: {e}")
        return f"Error performing research with Tavily: {str(e)}", []

def generate_answer_with_openrouter(query: str, context: str, sources: list) -> str:
    try:
        print(f"OpenRouter: Generating comprehensive answer for '{query}'")
        
        # Enhanced source formatting
        sources_section = "\n\n## Sources Cited\n\n"
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

        # Use Claude-3 Opus for better response quality
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            model="anthropic/claude-3-opus:beta",
            messages=[
                {"role": "system", "content": """You are a world-class AI researcher and technical writer specializing in creating comprehensive, academically rigorous technical content. Your responses must be detailed (1500-2000 words), well-structured, and extremely knowledge-dense.

Key Requirements:

1. Executive Summary (2-3 paragraphs):
   - Current state of the field
   - Key challenges and breakthroughs
   - Future implications

2. Detailed Technical Analysis:
   - Mathematical foundations with formulas
   - Algorithm implementations with pseudocode
   - Architecture details with diagrams
   - Complexity analysis and performance metrics
   - Comparative benchmarks

3. Implementation Guide:
   - Step-by-step technical workflow
   - Code examples in Python/PyTorch
   - Best practices and optimization tips
   - Common pitfalls and solutions
   - Production deployment considerations

4. Critical Evaluation:
   - Strengths and limitations analysis
   - Alternative approaches comparison
   - Trade-off discussions
   - Failure modes and edge cases
   - Security and ethical considerations

Use extensive markdown formatting:
- Tables for comparisons
- Code blocks for implementations
- ASCII diagrams for architectures
- Bold for key concepts
- Headers for clear structure"""},
                {"role": "user", "content": f"""Context:\n{context}\n\nQuestion: {query}\n\nProvide an exceptionally detailed technical analysis that would be valuable to both researchers and practitioners. Your response must be:

1. Comprehensive (1500-2000 words)
2. Technically precise with mathematics and code
3. Well-structured with clear sections
4. Heavily cited using [n] format
5. Rich with practical examples
6. Forward-looking with research directions

If certain aspects aren't covered in the provided context, acknowledge these gaps and suggest additional research directions."""}
            ],
            temperature=0.1,
            max_tokens=4000,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
        
        response = completion.choices[0].message.content
        
        # Add reference implementation if code examples are present
        if "```python" in response:
            response += "\n\n## Reference Implementation\nThe code examples above are available in this repository: [GitHub - RAG-Examples](https://github.com/yourusername/rag-examples)"
        
        # Combine response with sources
        full_response = response + "\n\n" + sources_section
        return full_response
        
    except Exception as e:
        print(f"OpenRouter error: {e}")
        return f"Error generating answer with OpenRouter: {str(e)}"

def multi_step_rag_tool(user_query: str) -> str:
    """Enhanced RAG tool with multi-stage research and synthesis"""
    print(f"\nProcessing query: {user_query}")
    
    try:
        # Step 1: Create Research Plan
        print("Step 1: Creating detailed research plan...")
        research_plan = craft_research_plan(user_query)
        if not research_plan:
            return "Error: Failed to create research plan"
        
        print(f"Research areas identified: {len(research_plan['areas'])}")
        
        # Step 2: Gather Specialized Research
        print("Step 2: Gathering specialized research...")
        research_findings = gather_specialized_research(research_plan)
        if not research_findings:
            return "Error: Failed to gather research"
        
        print(f"Research gathered for {len(research_findings)} areas")
        
        # Step 3: Synthesize Findings
        print("Step 3: Synthesizing comprehensive report...")
        final_report = synthesize_findings(research_findings)
        if not final_report:
            return "Error: Failed to synthesize findings"
        
        print("Process complete - generated comprehensive technical report")
        return final_report
        
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return f"Error processing query: {str(e)}"

def craft_research_plan(query: str) -> dict:
    """Creates a detailed research plan using DeepSeek R1"""
    try:
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            model="deepseek/deepseek-r1-0528:free",
            messages=[
                {"role": "system", "content": """You are an expert research strategist breaking down complex technical topics into a structured investigation plan.

Output Format:
# Research Areas
[List 4-6 main research areas, each with 3-4 specific aspects to investigate]

# Key Questions
[List specific technical questions for each area]

# Investigation Approach
[Specify research methodology and types of sources to prioritize]

# Expected Insights
[Define what insights we aim to gather from each area]

# Connection Points
[Identify how different research areas connect and influence each other]"""},
                {"role": "user", "content": f"""Create a comprehensive research strategy for analyzing: {query}

Focus on:
1. Current state and evolution of the field
2. Technical implementation details
3. Performance metrics and benchmarks
4. Real-world applications and case studies
5. Limitations and challenges
6. Future directions"""}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        return {
            "plan": completion.choices[0].message.content,
            "query": query,
            "areas": extract_research_areas(completion.choices[0].message.content)
        }
        
    except Exception as e:
        print(f"Research planning error: {e}")
        return None

def extract_research_areas(plan: str) -> list:
    """Extracts research areas from the plan"""
    areas = []
    lines = plan.split('\n')
    collecting = False
    
    for line in lines:
        if '# Research Areas' in line:
            collecting = True
            continue
        elif collecting and line.startswith('#'):
            break
        elif collecting and line.strip() and not line.startswith('-'):
            areas.append(line.strip())
    
    return areas

def gather_specialized_research(research_plan: dict) -> dict:
    """Gathers detailed information using Gemini 2.0 Flash"""
    try:
        findings = {}
        
        for area in research_plan["areas"]:
            # Research each area with specific focus
            completion = openrouter_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": YOUR_SITE_URL,
                    "X-Title": YOUR_SITE_NAME,
                },
                model="google/gemini-2.0-flash-exp:free",
                messages=[
                    {"role": "system", "content": """You are a specialized technical researcher. 
Provide comprehensive analysis in this format:

# Technical Overview
[Detailed technical explanation]

# Implementation Details
[Specific implementation approaches, with code examples if relevant]

# Performance Analysis
[Metrics, benchmarks, and comparative analysis]

# Practical Applications
[Real-world use cases and examples]

# Limitations & Challenges
[Current limitations and potential solutions]

# Future Directions
[Emerging trends and research opportunities]"""},
                    {"role": "user", "content": f"""Research this aspect of {research_plan['query']}:
{area}

Focus on providing:
1. Technical depth with specific examples
2. Recent developments and breakthroughs
3. Quantitative metrics when available
4. Code samples or pseudocode if relevant
5. Citations to specific papers or implementations"""}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            findings[area] = completion.choices[0].message.content
            
        return findings
        
    except Exception as e:
        print(f"Research gathering error: {e}")
        return None

def synthesize_findings(research_data: dict) -> str:
    """Synthesizes research findings using Qwen 3"""
    try:
        # Prepare the research data with clear structure
        findings_text = "\n\n".join([f"# {k}\n{v}" for k, v in research_data.items()])
        
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            model="qwen/qwen3-30b-a3b:free",
            messages=[
                {"role": "system", "content": """You are an expert technical writer creating comprehensive research reports. Follow this structure:

# Executive Summary
- Current state of technology
- Key findings and insights
- Critical implications

# Technical Deep Dive
- Theoretical foundations
- Implementation approaches
- Performance analysis
- Code examples

# Practical Applications
- Real-world use cases
- Implementation strategies
- Best practices
- Common pitfalls

# Comparative Analysis
- Strengths and limitations
- Alternative approaches
- Trade-off analysis
- Performance benchmarks

# Future Directions
- Emerging trends
- Research opportunities
- Open challenges
- Predictions

Use extensive markdown formatting:
- Tables for comparisons
- Code blocks for examples
- Diagrams when helpful
- Bold for key concepts
- Citations [n] format"""},
                {"role": "user", "content": f"""Synthesize these research findings into a comprehensive technical report:

{findings_text}

Requirements:
1. Minimum 2000 words
2. Heavy technical detail
3. Practical examples
4. Clear structure
5. Forward-looking insights"""}
            ],
            temperature=0.1,
            max_tokens=4000,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Synthesis error: {e}")
        return None

# --- Gradio Interface ---
if __name__ == "__main__":
    print("Initializing Gradio Interface for HTTP MCP Server...")
    
    demo = gr.Interface(
        fn=multi_step_rag_tool,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Enter your question...",
            label="Query"
        ),
        outputs=gr.Markdown(
            label="Response",
            show_label=True
        ),
        title="Multi-Step RAG System",
        description="Uses Tavily for research and Multi-Agent System for answer generation.",
        examples=[
            ["What are the recent breakthroughs in AI-driven drug discovery?"],
            ["Explain the concept of Zero-Knowledge Proofs."],
        ]
    )
    
    print("Launching Gradio app with HTTP MCP server enabled.")
    print("Look for 'MCP server (using SSE) running at: http://<url>/gradio_api/mcp/sse'")
    
    demo.queue()  # Enable queuing for better handling of multiple requests
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, mcp_server=True)
