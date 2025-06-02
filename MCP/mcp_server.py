# app.py (HTTP SSE Gradio MCP Server)

import os
import gradio as gr
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from logger_config import setup_logging
from utils import validate_response, parse_research_results, format_sources_section

# Set up logging
loggers = setup_logging()
server_logger = loggers['server']
research_logger = loggers['research']
synthesis_logger = loggers['synthesis']

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:7860")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "My RAG MCP Server")

# Verify required environment variables
required_vars = {
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "TAVILY_API_KEY": TAVILY_API_KEY,
    "YOUR_SITE_URL": YOUR_SITE_URL,
    "YOUR_SITE_NAME": YOUR_SITE_NAME
}

missing_vars = [key for key, value in required_vars.items() if not value]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    server_logger.error(error_msg)
    raise EnvironmentError(f"{error_msg}\nPlease check your .env file and ensure all variables are set.")

# Enhanced configuration
MODELS = {
    "planner": "deepseek/deepseek-r1-0528:free",
    "researcher": "deepseek/deepseek-chat-v3-0324:free", 
    "synthesizer": "qwen/qwen3-30b-a3b:free"
}

# Initialize clients with proper configuration
try:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": YOUR_SITE_NAME
        }
    )
    server_logger.info("OpenRouter client initialized successfully")
except Exception as e:
    server_logger.error(f"Failed to initialize OpenRouter client: {str(e)}")
    raise

try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    server_logger.info("Tavily client initialized successfully")
except Exception as e:
    server_logger.error(f"Failed to initialize Tavily client: {str(e)}")
    raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def research_with_tavily(query: str, max_results: int = 12) -> tuple[str, list]:
    try:
        research_logger.info(f"Starting research for query: {query}")
        
        if not query.strip():
            raise ValueError("Empty query provided")
        
        research_logger.debug("Sending search request to Tavily API")
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=[
                "arxiv.org", "papers.ssrn.com", "github.com", 
                "deepseek.ai", "huggingface.co", "paperswithcode.com"
            ],
            include_answer=True,
            search_type="comprehensive"
        )
        
        research_logger.debug("Validating Tavily API response")
        if not validate_response(response, dict):
            raise ValueError(f"Invalid response type from Tavily API: {type(response)}")
        
        if "results" not in response:
            raise ValueError("No results field in Tavily API response")
        
        results = response["results"]
        if not validate_response(results, list):
            raise ValueError(f"Invalid results type from Tavily API: {type(results)}")
        
        if not results:
            research_logger.warning("No results found in research phase")
            return "No relevant information found in the research phase.", []
        
        research_logger.debug("Parsing research results")
        contexts, sources = parse_research_results(results)
        
        if not contexts:
            research_logger.warning("Found results but no valid content could be extracted")
            return "Found results but no valid content could be extracted.", []
        
        research_logger.info(
            f"Research complete: {len(contexts)} sources found "
            f"({len([s for s in sources if s['type'] == 'research_paper'])} research papers)"
        )
        return "\n\n".join(contexts), sources
            
    except Exception as e:
        error_msg = f"Error performing research with Tavily: {str(e)}"
        research_logger.error(error_msg, exc_info=True)
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_answer_with_openrouter(query: str, context: str, sources: list) -> str:
    try:
        synthesis_logger.info(f"Starting answer generation for query: {query}")
        
        if not query.strip():
            raise ValueError("Empty query provided")
            
        if not context.strip():
            raise ValueError("No research context provided")
        
        synthesis_logger.debug("Preparing sources section")
        sources_section = format_sources_section(sources)
        
        try:
            synthesis_logger.debug("Sending completion request to OpenRouter API")
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
            
            if not completion or not completion.choices:
                raise ValueError("Empty completion response from OpenRouter API")
            
            response = completion.choices[0].message.content
            if not response:
                raise ValueError("Empty response content from language model")
            
            synthesis_logger.debug("Processing and formatting response")
            
            # Add reference implementation if code examples are present
            if "```python" in response:
                response += "\n\n## Reference Implementation\nThe code examples above are available in this repository: [GitHub - RAG-Examples](https://github.com/yourusername/rag-examples)"
            
            # Combine response with sources
            full_response = response + "\n\n" + sources_section
            
            synthesis_logger.info("Successfully generated comprehensive response")
            return full_response
            
        except Exception as e:
            raise Exception(f"Error generating response with language model: {str(e)}")
            
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        synthesis_logger.error(error_msg, exc_info=True)
        raise

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

def process_query(query: str) -> str:
    """Main function to process user queries through the multi-step RAG system"""
    try:
        server_logger.info(f"Processing new query: {query}")
        
        # Step 1: Research Phase
        server_logger.debug("Starting research phase")
        try:
            context, sources = research_with_tavily(query)
            if "Error performing research" in context:
                raise Exception(context)
        except Exception as e:
            server_logger.error(f"Research phase failed: {str(e)}", exc_info=True)
            return f"Failed to gather research: {str(e)}"
            
        # Step 2: Synthesis Phase
        server_logger.debug("Starting synthesis phase")
        try:
            response = generate_answer_with_openrouter(query, context, sources)
            if "Error generating answer" in response:
                raise Exception(response)
        except Exception as e:
            server_logger.error(f"Synthesis phase failed: {str(e)}", exc_info=True)
            return f"Failed to generate response: {str(e)}"
            
        server_logger.info("Query processing completed successfully")
        return response
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        server_logger.error(error_msg, exc_info=True)
        return error_msg

# Create Gradio interface
server_logger.info("Initializing Gradio interface")
iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter your research query here...",
        label="Research Query"
    ),
    outputs=gr.Markdown(
        label="Research Results",
        show_label=True
    ),
    title="Multi-Step RAG Research System",
    description="""This system performs deep technical research using a multi-step process:
1. Gathers comprehensive research from academic and technical sources
2. Synthesizes findings into a detailed technical analysis
3. Provides implementation guidance and practical examples""",
    examples=[
        ["What are the latest advances in transformer architecture optimizations?"],
        ["Explain the mathematical foundations of diffusion models"],
        ["Compare and analyze different approaches to few-shot learning"]
    ],
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    try:
        server_logger.info(f"Starting Gradio server at {YOUR_SITE_URL}")
        iface.launch(
            server_name="0.0.0.0",
            share=False,
            debug=True
        )
    except Exception as e:
        server_logger.error(f"Failed to start Gradio server: {str(e)}", exc_info=True)
        raise
