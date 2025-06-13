# multi_agents/main.py

import asyncio
import json
import logging
import os
from typing import Dict, Any, Callable, Coroutine
from concurrent.futures import Future # Needed for run_coroutine_threadsafe

from .agents_logic import OrchestratorAgent, ReportAgent, PlannerAgent
from utils.research_utils import parse_research_results, format_sources_section
from tavily import TavilyClient
import google.generativeai as genai

# Load API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

logger = logging.getLogger(__name__)

# --- 1. The synchronous task function now accepts the event loop ---
def _synchronous_research_task(
    query: str,
    stream_output_callback: Callable[[str], Coroutine],
    # --- FIX 1: Accept the event loop as an argument ---
    loop: asyncio.AbstractEventLoop
) -> Dict[str, Any]:
    """
    This function contains your entire working research logic.
    It is synchronous and designed to be run in a separate thread.
    It now accepts a callback and the main event loop to report progress.
    """
    # This helper function allows our synchronous thread to safely call the async callback
    def report_progress(message: str):
        # --- FIX 2: Use the passed-in loop to schedule the coroutine ---
        # run_coroutine_threadsafe returns a Future, we don't need to await it here.
        # We just need to schedule it safely from this thread to the target loop.
        try:
            asyncio.run_coroutine_threadsafe(stream_output_callback(message), loop)
        except Exception as e:
            # Log potential errors in scheduling the callback
            logger.error(f"Failed to schedule progress callback: {e}")

    try:
        if not GEMINI_API_KEY or not TAVILY_API_KEY:
            raise ValueError("GEMINI_API_KEY and TAVILY_API_KEY must be set in the .env file.")

        report_progress("[Worker] Starting deep research task...")
        
        genai.configure(api_key=GEMINI_API_KEY)
        orchestrator = OrchestratorAgent(use_gemini=True, api_key=GEMINI_API_KEY)
        planner = PlannerAgent(use_gemini=True, api_key=GEMINI_API_KEY)
        report_agent = ReportAgent(use_gemini=True, api_key=GEMINI_API_KEY)
        
        report_progress("[Worker] Creating research plan...")
        research_plan = orchestrator.create_research_plan(query)

        all_search_results, search_count, seen_urls = [], 0, set()
        MAX_SEARCHES_TOTAL = 15
        
        while search_count < MAX_SEARCHES_TOTAL:
            current_contexts = [r['content'] for r in all_search_results if r.get('content')]
            progress = orchestrator.evaluate_research_progress(research_plan, current_contexts)
            if all(progress.values()):
                report_progress("[Worker] All research aspects covered.")
                break
            
            remaining_items = planner.prioritize_unfulfilled_requirements(research_plan, progress, current_contexts)
            if not remaining_items:
                report_progress("[Worker] No more unfulfilled items to research.")
                break

            item_type, research_item = remaining_items[0]
            report_progress(f"[Worker] Researching: {item_type} - '{research_item}'")
            search_queries = planner.create_search_strategy(research_item, item_type)

            for sq in search_queries:
                if search_count >= MAX_SEARCHES_TOTAL: break
                report_progress(f"[Worker] Searching for: '{sq}'")
                
                try:
                    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
                    results = tavily_client.search(query=sq, search_depth="advanced", max_results=5)
                except Exception as search_error:
                    report_progress(f"[Worker] WARNING: Tavily search failed for query '{sq}': {search_error}")
                    continue

                search_count += 1
                for result in results.get('results', []):
                    if result.get('url') and result.get('url') not in seen_urls:
                        all_search_results.append(result)
                        seen_urls.add(result.get('url'))
        
        # --- The critical check from before remains ---
        if not all_search_results:
            report_progress("[Worker] ERROR: Research complete, but no relevant sources were found.")
            raise ValueError("No relevant search results were found for the query. Cannot generate a report.")
        # --------------------------------------------

        report_progress(f"[Worker] Found {len(all_search_results)} sources. Generating final report...")
        contexts, sources = parse_research_results(all_search_results)
        completion_stats = {"total_searches": search_count, "unique_sources": len(seen_urls)}
        report_text = report_agent.generate_report(query, research_plan, contexts, completion_stats)
        report_text += "\n\n" + format_sources_section(sources)
        
        report_progress("[Worker] Task finished successfully.")
        return {
            "status": "success",
            "report": report_text,
        }

    except Exception as e:
        logger.error(f"[Worker] Deep research failed: {str(e)}", exc_info=True)
        report_progress(f"[Worker] ERROR: Research task failed critically: {e}")
        raise # Re-raise the exception

# --- 2. The async entry point gets the loop and passes it ---
async def run_research_task(query: str, stream_output_callback: Callable[[str], Coroutine]) -> Dict[str, Any]:
    # --- FIX 3: Get the current running loop here (in the main async thread) ---
    loop = asyncio.get_running_loop()
    # ---------------------------------------------------------------------
    
    # --- FIX 4: Pass the loop object to the synchronous task function ---
    return await asyncio.to_thread(
        _synchronous_research_task,
        query,
        stream_output_callback,
        loop # Pass the loop
    )
    # ------------------------------------------------------------------

# The main execution block (if __name__ == "__main__") is not needed here.
# It is in mcp_server.py and llm_tool_caller_client.py