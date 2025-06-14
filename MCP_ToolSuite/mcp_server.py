# mcp_server.py

import asyncio
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging
import json
import uvicorn

# New imports for new tools
import matplotlib.pyplot as plt
import numpy as np

from mcp.server.fastmcp import FastMCP, Context

# This is our bridge file that contains the core logic for the research tool
from multi_agents.main import run_research_task

# --- 1. SETUP AND CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories for generated files if they don't exist
os.makedirs("generated_charts", exist_ok=True)


# --- 2. THE MCP SERVER AND TOOLS ---
mcp = FastMCP("MultiToolAgent")

# --- A helper function for the new "Query Optimizer" logic ---
async def get_optimized_search_queries(original_query: str, num_queries: int = 2) -> list[str]:
    # This helper is now even more important
    from multi_agents.agents_logic import BaseAgent
    agent = BaseAgent(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"You are a search query optimization expert. Distill the user's request into {num_queries} simple, effective, keyword-rich search queries for a web search engine. User Request: \"{original_query}\". Respond with ONLY a valid JSON array of strings."
    try:
        response_text = agent.generate(prompt, "")
        queries = json.loads(response_text.strip().replace("```json", "").replace("```", ""))
        logger.info(f"[Query Optimizer] For '{original_query[:50]}...' -> Generated Queries: {queries}")
        return queries
    except Exception:
        return [original_query]


@mcp.tool()
async def quick_search(query: str, ctx: Context = None) -> Dict[str, str]:
    """
    Use for single, factual questions that need a fast, real-time web search. Ideal for: 'What is the weather in Hanoi?', 'Who won the last World Cup?', 'What is the stock price of Apple?'.
    """
    from tavily import TavilyClient
    from multi_agents.agents_logic import BaseAgent
    logger.info(f"MCP Tool: ADVANCED search for '{query}'")
    
    agent = BaseAgent(api_key=os.getenv("GEMINI_API_KEY"))
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    current_query = query
    for attempt in range(2): # Try up to 2 times (initial + 1 retry)
        logger.info(f"Search attempt {attempt + 1} with query: '{current_query}'")
        try:
            # Step 1: Search
            search_results = tavily.search(query=current_query, search_depth="advanced", max_results=5)
            context = "\n---\n".join([res.get('content', '') for res in search_results.get('results', [])])

            if not context:
                logger.warning("Search returned no content.")
                continue

            # Step 2: Evaluate if the answer is in the context
            eval_prompt = f"""
            Based on the following text, can you find the specific answer for the user's original question?
            Original Question: "{query}"
            Text to search:
            ---
            {context}
            ---
            If you find a direct answer, state it clearly. If the answer is not in the text, respond with ONLY the word "null".
            """
            evaluation = agent.generate(eval_prompt, "")
            
            # Step 3: Check evaluation and return if successful
            if "null" not in evaluation.lower():
                logger.info("Found a definitive answer.")
                return {"status": "success", "answer": evaluation}

            # Step 4: If failed, refine the query for the next attempt
            logger.info("Answer not found in results, refining search query.")
            refine_prompt = f"""
            My web search for "{current_query}" did not yield a direct answer to the user's question: "{query}".
            Generate one new, more specific search query to try next. Think about what a human expert would search for.
            For example, if the query is about a stock price, add the ticker symbol (e.g., 'NASDAQ:NVDA').
            Respond with ONLY the new search query text, and nothing else.
            """
            current_query = agent.generate(refine_prompt, "")
            
        except Exception as e:
            logger.error(f"An error occurred during search attempt {attempt + 1}: {e}")

    logger.warning("Failed to find a definitive answer after multiple attempts.")
    return {"status": "success", "answer": "I performed several searches but could not find a definitive answer to your query."}

@mcp.tool()
async def medium_search(query: str, ctx: Context = None) -> Dict[str, str]:
    """
    Use for broader questions that require generating lists, recommendations, or comparisons from several web sources. Ideal for: 'What are some fun indoor activities in Hanoi for a rainy day?', 'Compare React vs. Vue for web development'.
    """
    from tavily import TavilyClient
    from multi_agents.agents_logic import BaseAgent
    logger.info(f"MCP Tool: MEDIUM search for '{query}'")
    try:
        optimized_queries = await get_optimized_search_queries(query, num_queries=3)
        
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        all_results_content = []
        for opt_query in optimized_queries:
            search_results = tavily.search(query=opt_query, search_depth="advanced", max_results=3)
            all_results_content.extend([res.get('content', '') for res in search_results.get('results', [])])
        
        context = "\n---\n".join(filter(None, all_results_content))

        if not context:
            return {"status": "success", "answer": "I performed several searches but could not find any relevant information for your query."}

        agent = BaseAgent(api_key=os.getenv("GEMINI_API_KEY"))
        synthesis_prompt = f"Based ONLY on the following search results, provide a helpful, well-structured answer to the user's original question. If the user asks for a list, use bullet points.\n\nOriginal Question: {query}\n\nSearch Results:\n{context}"
        answer = agent.generate(synthesis_prompt, "")
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "answer": f"Failed to perform medium search: {e}"}

@mcp.tool()
async def conduct_deep_research(query: str, ctx: Context = None) -> Dict[str, Any]:
    """
    Use only for complex, in-depth queries that require a formal, multi-page report with deep analysis, planning, and multiple research steps.
    Ideal for: 'A detailed report on the architecture of V-JEPA 2', 'Analyze the economic impact of quantum computing'.
    Do NOT use for simple questions or lists. This tool is slow and expensive.
    """
    try:
        logger.info(f"MCP Tool: Received request for DEEP research on '{query}'")
        async def stream_progress_to_client(log_message: str):
            if ctx:
                await ctx.info(log_message)
                logger.info(f"[Progress Stream to Client] {log_message}")

        final_report_dict = await run_research_task(
            query=query,
            stream_output_callback=stream_progress_to_client
        )
        logger.info("MCP Tool: Deep research task completed.")
        return final_report_dict
    except Exception as e:
        logger.error(f"MCP Tool: Error during deep research: {e}", exc_info=True)
        return {"status": "error", "error_message": str(e), "report": ""}


@mcp.tool()
async def summarize_text(text_to_summarize: str, format: str, ctx: Context = None) -> Dict[str, str]:
    """
    Condenses a long piece of text into a summary.
    The 'format' can be 'a single paragraph' or 'bullet points'.
    """
    from multi_agents.agents_logic import BaseAgent
    logger.info(f"MCP Tool: Received request to SUMMARIZE text.")
    try:
        agent = BaseAgent(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = f"Please summarize the following text into {format}:\n\n---TEXT---\n{text_to_summarize}\n---END TEXT---"
        system_prompt = "You are an expert summarizer. You follow the user's formatting instructions precisely."
        summary = agent.generate(prompt, system_prompt)
        return {"status": "success", "summary": summary}
    except Exception as e:
        logger.error(f"MCP Tool: Error during summarization: {e}", exc_info=True)
        return {"status": "error", "summary": f"Failed to summarize: {e}"}


@mcp.tool()
async def visualize_data(
    chart_type: str, 
    data: Dict[str, Any],  # FIX 1: Accept a Dictionary directly, not a string
    title: str, 
    x_label: Optional[str] = None, 
    y_label: Optional[str] = None, 
    ctx: Context = None
) -> Dict[str, str]:
    """
    Generates a data visualization (bar, line, or pie chart) from a JSON object and saves it as an image. Returns the file path.
    """
    logger.info(f"MCP Tool: VISUALIZE data as a {chart_type} chart.")
    try:
        # FIX 2: No need for json.loads(), as 'data' is already a dictionary
        plt.figure()

        if chart_type == "bar":
            plt.bar(data['labels'], data['values'])
            if x_label: plt.xlabel(x_label)
            if y_label: plt.ylabel(y_label)
        elif chart_type == "line":
            plt.plot(data['labels'], data['values'])
            if x_label: plt.xlabel(x_label)
            if y_label: plt.ylabel(y_label)
        elif chart_type == "pie":
            plt.pie(data['sizes'], labels=data['labels'], autopct='%1.1f%%', startangle=90)
        else:
            return {"status": "error", "message": f"Unsupported chart type: {chart_type}"}
        
        plt.title(title)
        plt.tight_layout()
        
        timestamp = asyncio.get_running_loop().time()
        file_path = os.path.join("generated_charts", f"chart_{int(timestamp)}.png")
        plt.savefig(file_path)
        plt.close()
        
        return {"status": "success", "message": f"Chart saved to {file_path}", "image_path": file_path}
    except Exception as e:
        return {"status": "error", "message": f"Failed to create chart: {e}"}


@mcp.tool()
async def explain_code(code_snippet: str, programming_language: str, ctx: Context = None) -> Dict[str, str]:
    """
    Analyzes a snippet of code and provides a structured, easy-to-understand explanation for a non-technical audience. It includes a purpose, an analogy, inputs/outputs, and potential risks.
    """
    from multi_agents.agents_logic import BaseAgent
    logger.info(f"MCP Tool: Received request to EXPLAIN {programming_language} code.")
    try:
        agent = BaseAgent(api_key=os.getenv("GEMINI_API_KEY"))
        
        prompt = f"""
        Analyze the following {programming_language} code snippet and generate a structured explanation for a non-technical manager.

        Code:
        ```
        {code_snippet}
        ```

        Your response must be a markdown-formatted report with the following four sections exactly:

        ## 🎯 Purpose
        (A one-sentence summary of what this code achieves from a business or user perspective.)

        ##  analogies
        (A simple, real-world analogy to explain the core concept of the code. For example: "Think of this code as an automated mail sorter that...")

        ## 📥 Inputs & 📤 Outputs
        (Clearly state what information the code NEEDS to run and what it PRODUCES as a result.)
        - **Inputs:** ...
        - **Outputs:** ...

        ## ⚠️ Potential Risks & Questions
        (List 2-3 important questions a manager should ask their developers about this code. For example: "Is this code secure?", "How does it handle errors?", "Where does the data come from?")
        """
        system_prompt = "You are an expert software developer who is brilliant at explaining complex code to non-technical stakeholders using structured reports and simple analogies."
        explanation = agent.generate(prompt, system_prompt)
        return {"status": "success", "explanation": explanation}
    except Exception as e:
        return {"status": "error", "explanation": f"Failed to explain code: {e}"}



# --- 3. SERVER STARTUP ---
if __name__ == "__main__":
    logger.info("Starting Multi-Tool Agent MCP Server...")
    mcp.run(transport='sse')