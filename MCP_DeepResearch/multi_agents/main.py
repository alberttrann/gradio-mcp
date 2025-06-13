# MCP_DeepResearch/multi_agents/main.py

import asyncio
import json
import logging
import os
from typing import Dict, Any, Callable, Coroutine

from .agents_logic import OrchestratorAgent, ReportAgent, PlannerAgent
from utils.research_utils import parse_research_results, format_sources_section
from tavily import TavilyClient

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

logger = logging.getLogger(__name__)

def _synchronous_research_task(
    query: str,
    stream_output_callback: Callable[[str], Coroutine],
    loop: asyncio.AbstractEventLoop
) -> Dict[str, Any]:
    def report_progress(message: str):
        try:
            asyncio.run_coroutine_threadsafe(stream_output_callback(f"[Worker] {message}"), loop)
        except Exception as e:
            logger.error(f"Failed to schedule progress callback: {e}")

    try:
        if not GEMINI_API_KEY or not TAVILY_API_KEY:
            raise ValueError("GEMINI_API_KEY and TAVILY_API_KEY must be set.")

        report_progress("Initializing agent swarm...")
        api_key_config = {"api_key": GEMINI_API_KEY}
        orchestrator = OrchestratorAgent(**api_key_config)
        planner = PlannerAgent(**api_key_config)
        report_agent = ReportAgent(**api_key_config)
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

        report_progress("Creating strategic research plan...")
        research_plan = orchestrator.create_research_plan(query)

        all_search_results = []
        seen_urls = set()
        search_count = 0
        MAX_SEARCHES_TOTAL = 15
        research_attempts = {}
        MAX_ATTEMPTS_PER_ITEM = 2

        while search_count < MAX_SEARCHES_TOTAL:
            # --- THIS IS THE FIX ---
            # We must only evaluate progress AFTER we have some search results.
            # On the first pass, we skip evaluation and go straight to prioritizing and searching.
            
            current_contexts = [res.get('content', '') for res in all_search_results]

            if all_search_results: # This check prevents evaluation on the first empty run
                progress = orchestrator.evaluate_research_progress(research_plan, current_contexts)
                if all(progress.values()):
                    report_progress("Research evaluation complete. All objectives met.")
                    break
            else:
                # On the first run, progress is definitionally false
                progress = {"core_concepts": False, "key_questions": False, "information_requirements": False}
            
            # --- END OF FIX ---

            remaining_items = planner.prioritize_unfulfilled_requirements(research_plan, progress, current_contexts)

            if not remaining_items:
                report_progress("No more research items to prioritize. Concluding research phase.")
                break
            
            next_item_to_research = None
            for item_type, research_item in remaining_items:
                item_key = f"{item_type}:{research_item}"
                if research_attempts.get(item_key, 0) < MAX_ATTEMPTS_PER_ITEM:
                    next_item_to_research = (item_type, research_item)
                    break 
            
            if not next_item_to_research:
                report_progress("All remaining research items have reached max attempts. Concluding research.")
                break
            
            item_type, research_item = next_item_to_research
            item_key = f"{item_type}:{research_item}"
            research_attempts[item_key] = research_attempts.get(item_key, 0) + 1

            report_progress(f"Deep diving into: {item_type} - '{research_item}' (Attempt {research_attempts[item_key]})")
            
            search_queries = planner.create_search_strategy(research_item, item_type)
            
            for sq in search_queries:
                if search_count >= MAX_SEARCHES_TOTAL: break
                report_progress(f"Searching for: '{sq}'")
                try:
                    results = tavily_client.search(query=sq, search_depth="advanced", max_results=5)
                    for result in results.get('results', []):
                        if result.get('url') and result['url'] not in seen_urls:
                            all_search_results.append(result)
                            seen_urls.add(result['url'])
                    search_count += 1
                except Exception as search_error:
                    report_progress(f"WARNING: Tavily search failed for '{sq}': {search_error}")

        if not all_search_results:
            raise ValueError("Research concluded with no relevant sources found.")

        report_progress(f"Found {len(seen_urls)} unique sources. Generating final report...")
        contexts, sources = parse_research_results(all_search_results)
        
        completion_stats = {
            "total_searches_conducted": search_count,
            "unique_sources_found": len(seen_urls),
            "final_research_coverage": progress
        }
        
        report_text = report_agent.generate_report(query, research_plan, contexts, completion_stats)
        report_text += format_sources_section(sources)
        
        report_progress("Task finished successfully.")
        return {"status": "success", "report": report_text}

    except Exception as e:
        logger.error(f"[Worker] Deep research failed critically: {str(e)}", exc_info=True)
        report_progress(f"ERROR: Research task failed critically: {e}")
        raise

async def run_research_task(query: str, stream_output_callback: Callable[[str], Coroutine]) -> Dict[str, Any]:
    """Async entry point that runs the synchronous research task in a separate thread."""
    loop = asyncio.get_running_loop()
    return await asyncio.to_thread(
        _synchronous_research_task,
        query,
        stream_output_callback,
        loop
    )