# mcp_server.py

import asyncio
from typing import Dict, Any
import os
from dotenv import load_dotenv
import logging
import json
import uvicorn

from mcp.server.fastmcp import FastMCP, Context

# This is our bridge file that contains the core logic
from multi_agents.main import run_research_task

# --- 1. SETUP AND CONFIGURATION ---
load_dotenv()
# Logger Setup - Ensure this is at the top level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Create a clean, simple MCP server
mcp = FastMCP("DeepResearchAgent")

@mcp.tool()
async def conduct_deep_research(
    query: str,
    ctx: Context = None
) -> Dict[Any, Any]:
    """
    Performs comprehensive, multi-agent research on a given query.
    Use this for complex questions that require web searches, planning, and a detailed final report.
    """
    try:
        logger.info(f"MCP Tool: Received request for deep research on '{query}'")

        async def stream_progress_to_client(log_message: str):
            if ctx:
                await ctx.info(log_message)
                # Also log to the server's console for easier debugging
                logger.info(f"[Progress Stream to Client] {log_message}")


        final_report_dict = await run_research_task(
            query=query,
            stream_output_callback=stream_progress_to_client
        )

        if final_report_dict.get("status") == "success" and not final_report_dict.get("report", "").strip():
            logger.warning("Research task completed, but the final report is empty.")
            return {
                "status": "error",
                "error_message": "Research completed, but the final report was empty. This might happen if no relevant sources were found.",
                "report": ""
            }

        logger.info("MCP Tool: Research task completed. Returning final dictionary.")
        return final_report_dict

    except ValueError as ve:
        logger.error(f"MCP Tool: ValueError during research: {ve}")
        if ctx:
            await ctx.error(f"Research failed: {ve}")
        return {
            "status": "error",
            "error_message": str(ve),
            "report": f"Research could not be completed: {ve}"
        }
    except Exception as e:
        logger.error(f"MCP Tool: An unexpected error occurred: {e}", exc_info=True)
        if ctx:
            await ctx.error(f"An unexpected server error occurred: {e}")
        return {
            "status": "error",
            "error_message": f"An unexpected server error occurred: {str(e)}",
            "report": ""
        }

# --- ORIGINAL SERVER STARTUP ---
if __name__ == "__main__":
    logger.info("Starting DeepResearchAgent MCP Server...")
    mcp.run(transport='sse')