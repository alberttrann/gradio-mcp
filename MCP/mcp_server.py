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

def research_with_tavily(query: str, max_results: int = 3) -> str:
    try:
        print(f"Tavily: Searching for '{query}'")
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        
        if response and response.get("results"):
            context = "\n\n".join([obj["content"] for obj in response["results"]])
            if not context.strip():
                return "No relevant information found by Tavily."
            print(f"Tavily: Found context (first 100 chars): {context[:100]}...")
            return context
        else:
            return "No relevant information found by Tavily"
    except Exception as e:
        print(f"Tavily error: {e}")
        return f"Error performing research with Tavily: {str(e)}"

def generate_answer_with_openrouter(query: str, context: str) -> str:
    try:
        print(f"OpenRouter: Generating answer for query '{query}'")
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            model="google/gemma-3-27b-it:free",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Answer based only on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the context:"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenRouter error: {e}")
        return f"Error generating answer with OpenRouter: {str(e)}"

def multi_step_rag_tool(user_query: str) -> str:
    """RAG tool that combines Tavily research with OpenRouter LLM responses"""
    print(f"\nReceived query: {user_query}")
    
    # Step 1: Research
    print("Step 1: Research with Tavily...")
    research_context = research_with_tavily(user_query)
    if "Error performing research" in research_context:
        return research_context
    
    # Step 2: Generate Answer
    print("Step 2: Generate answer with OpenRouter...")
    final_answer = generate_answer_with_openrouter(user_query, research_context)
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