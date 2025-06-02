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

def research_with_tavily(query: str, max_results: int = 8) -> tuple[str, list]:
    try:
        print(f"Tavily: Performing comprehensive research for '{query}'")
        
        # Expand search query to get more technical details
        expanded_query = f"""Detailed technical analysis of: {query}
        Include:
        - Technical implementation details
        - Real-world applications
        - Recent developments
        - Expert opinions
        - Comparative analysis
        - Best practices"""
        
        response = tavily_client.search(
            query=expanded_query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=[
                "arxiv.org", "papers.ssrn.com", "github.com", 
                "towardsdatascience.com", "medium.com", "distill.pub",
                "openai.com", "research.google", "microsoft.com/research",
                "paperswithcode.com", "huggingface.co/blog"
            ],
            include_answer=True,
            search_type="comprehensive"
        )
        
        if response and response.get("results"):
            contexts = []
            sources = []  # Track sources separately
            for result in response["results"]:
                title = result.get("title", "").strip()
                content = result.get("content", "").strip()
                url = result.get("url", "").strip()
                date = result.get("published_date", "").strip()
                
                if title and content:
                    # Store source metadata
                    sources.append({
                        "title": title,
                        "url": url,
                        "date": date if date else "Date not available"
                    })
                    
                    # Format content for context
                    source_block = f"""Content from: {title}

{content}"""
                    contexts.append(source_block)
            
            context = "\n\n---\n\n".join(contexts)
            print(f"Tavily: Found {len(contexts)} high-quality sources")
            return context, sources
        else:
            return "No relevant information found by Tavily", []
    except Exception as e:
        print(f"Tavily error: {e}")
        return f"Error performing research with Tavily: {str(e)}", []

def generate_answer_with_openrouter(query: str, context: str, sources: list) -> str:
    try:
        print(f"OpenRouter: Generating comprehensive answer for '{query}'")
        
        # Format sources for citation
        sources_section = "\n\n## Sources Cited\n\n"
        for idx, source in enumerate(sources, 1):
            sources_section += f"{idx}. [{source['title']}]({source['url']}) - {source['date']}\n"
        
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[
                {"role": "system", "content": """You are an expert AI research scientist specializing in providing comprehensive, academically rigorous answers. Your responses should be detailed, well-structured, and knowledge-dense, typically 1000-2000 words. Follow these requirements:

1. Document Structure:
   - Begin with a 2-3 paragraph executive summary
   - Include a detailed table of contents
   - Organize content into 4-6 major sections with subsections
   - Conclude with future directions and open challenges
   - Use consistent hierarchical headers (##, ###, ####)

2. Technical Depth:
   - Begin each section with core principles and theoretical foundations
   - Include mathematical formulas and algorithms where relevant
   - Explain implementation details and architectural considerations
   - Compare different methodologies with pros/cons analysis
   - Discuss computational complexity and performance implications
   - Include pseudocode or code examples for key concepts

3. Practical Applications:
   - Provide real-world case studies and examples
   - Include implementation guidelines and best practices
   - Discuss common challenges and their solutions
   - Share optimization strategies and techniques
   - Reference specific tools, frameworks, and libraries
   - Include deployment considerations

4. Research Context:
   - Cite relevant papers and sources using [n] format
   - Compare competing methodologies and approaches
   - Discuss experimental results and benchmarks
   - Acknowledge limitations and edge cases
   - Highlight recent developments and breakthroughs
   - Suggest promising research directions

5. Enhanced Formatting:
   - Use tables for feature/method comparisons
   - Create ASCII diagrams for complex concepts
   - Employ bullet points for key insights
   - Bold important terms and concepts
   - Use code blocks for technical examples
   - Include comparative tables

6. Synthesis and Insights:
   - Connect concepts across different sources
   - Provide critical analysis of methods
   - Highlight trade-offs and decision factors
   - Share expert recommendations
   - Discuss industry trends and adoption
   - Predict future developments"""},
                {"role": "user", "content": f"""Context:\n{context}\n\nQuestion: {query}\n\nProvide an in-depth, comprehensive analysis that demonstrates both theoretical understanding and practical expertise. Your response should be substantial (1000-2000 words) and knowledge-dense. Use proper markdown formatting. Cite sources using [n] format where n corresponds to the source number in the Sources Cited section. If certain aspects aren't covered in the context, acknowledge these gaps and suggest where to find more information.

Requirements:
1. Minimum 1000 words
2. Include all major sections from the system prompt
3. Use extensive citations
4. Include practical examples
5. Add comparison tables
6. Discuss limitations and future work"""}
            ],
            temperature=0.1,  # Lower temperature for more focused and detailed responses
            max_tokens=4000,  # Maximum token limit for longer responses
            presence_penalty=0.6,  # Encourage coverage of different aspects
            frequency_penalty=0.3  # Discourage repetition
        )
        
        # Combine the answer with the sources section
        full_response = completion.choices[0].message.content + sources_section
        return full_response
    except Exception as e:
        print(f"OpenRouter error: {e}")
        return f"Error generating answer with OpenRouter: {str(e)}"

def multi_step_rag_tool(user_query: str) -> str:
    """RAG tool that combines Tavily research with OpenRouter LLM responses"""
    print(f"\nReceived query: {user_query}")
    
    # Step 1: Research
    print("Step 1: Research with Tavily...")
    research_context, sources = research_with_tavily(user_query)
    if "Error performing research" in research_context:
        return research_context
    
    # Step 2: Generate Answer
    print("Step 2: Generate answer with OpenRouter...")
    final_answer = generate_answer_with_openrouter(user_query, research_context, sources)
    print("Process complete.")
    return final_answer

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
        description="Uses Tavily for research and Gemma 3 27B for answer generation.",
        examples=[
            ["What are the recent breakthroughs in AI-driven drug discovery?"],
            ["Explain the concept of Zero-Knowledge Proofs."],
        ]
    )
    
    print("Launching Gradio app with HTTP MCP server enabled.")
    print("Look for 'MCP server (using SSE) running at: http://<url>/gradio_api/mcp/sse'")
    
    demo.queue()  # Enable queuing for better handling of multiple requests
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, mcp_server=True)
