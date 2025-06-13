# MCP_DeepResearch/multi_agents/agents_logic.py

import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import logging
import json
import re

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, use_gemini: bool = True, api_key: Optional[str] = None, gemini_model: Optional[str] = None, **kwargs):
        self.use_gemini = use_gemini
        if not api_key:
            raise ValueError("API key is required.")
        genai.configure(api_key=api_key)
        # Use a more capable model for higher quality agent output
        self.gemini_model = gemini_model or "gemini-1.5-pro-latest"

    def generate(self, prompt: str, system_prompt: str) -> str:
        try:
            model = genai.GenerativeModel(model_name=self.gemini_model)
            # For Gemini, it's often better to structure the prompt with a clear system instruction
            # and then the user's detailed request.
            response = model.generate_content(
                [
                    f"System Prompt: {system_prompt}",
                    f"User Prompt: {prompt}"
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2  # Slightly increased temperature for more creative planning/writing
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise

class OrchestratorAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = """You are an expert research planner that develops comprehensive research strategies. Your role is to create structured research plans that identify what information is needed and why. Focus on the logical flow of information needed to answer the query comprehensively."""

    def create_research_plan(self, query: str) -> Dict[str, List[str]]:
        # This function is working well and remains unchanged.
        prompt = f"""Create a detailed research plan for the following query: '{query}'

        Return a JSON object with the following structure:
        {{
            "core_concepts": ["List of 2-3 fundamental concepts that need to be understood first."],
            "key_questions": ["List of 3-5 specific, deep questions that need to be answered to satisfy the user's query."],
            "information_requirements": ["List of specific technical details, data points, or comparisons needed to answer each question."],
            "research_priorities": ["An ordered list of the most important topics to start with."]
        }}

        Ensure the plan is logical, targeted, and designed to produce a deep, high-quality report."""
        response = self.generate(prompt, self.system_prompt)
        try:
            cleaned_response = response.strip().replace('```json', '').replace('```', '').strip()
            plan = json.loads(cleaned_response)
            logger.info(f"Generated research plan: {json.dumps(plan, indent=2)}")
            return plan
        except Exception:
            logger.error(f"Failed to parse research plan: {response}")
            return {"core_concepts": [query], "key_questions": [query], "information_requirements": [], "research_priorities": [query]}

    def evaluate_research_progress(self, plan: Dict[str, List[str]], gathered_info: List[str]) -> Dict[str, bool]:
        # --- THIS IS THE CRITICAL FIX FOR PARSING ---
        prompt = f"""Analyze the research plan and the gathered information to evaluate completeness.

        Research Plan:
        {json.dumps(plan, indent=2)}

        Gathered Information Summary (first 500 chars of each doc):
        {''.join([text[:500] + '...' for text in gathered_info])}

        Your task: Return a JSON object with boolean fields indicating if enough high-quality information has been found for each category.
        
        CRITICAL: Your response must contain ONLY the JSON object. Do not include markdown fences, "Rationale", or any other text.

        Required JSON format:
        {{
            "core_concepts": boolean,
            "key_questions": boolean,
            "information_requirements": boolean
        }}"""
        response = self.generate(prompt, self.system_prompt)
        try:
            # Use regex to find the JSON block, ignoring any extra text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in the response.")
            
            json_string = json_match.group(0)
            parsed = json.loads(json_string)
            
            # Basic validation
            required_keys = {"core_concepts", "key_questions", "information_requirements"}
            if not all(key in parsed for key in required_keys):
                 raise ValueError("Parsed JSON is missing required keys.")

            return parsed
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}\nFull Response:\n{response}")
            return {"core_concepts": False, "key_questions": False, "information_requirements": False}


class PlannerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = """You are an expert research planner that creates targeted, deep-dive search queries."""

    def create_search_strategy(self, research_item: str, item_type: str) -> List[str]:
        prompt = f"""Create 2-3 specific, advanced search queries for the following '{item_type}': "{research_item}"

        Focus on finding technical details, official documentation, academic papers, and expert analysis. Avoid high-level marketing content.

        Example for "Transformer Architecture":
        ["transformer model original paper 'Attention Is All You Need' pdf", "multi-head self-attention mechanism explained", "vision transformer (ViT) architecture details"]

        Return ONLY a JSON array of 2-3 carefully crafted search queries."""
        response = self.generate(prompt, self.system_prompt)
        try:
            cleaned_response = response.strip().replace('```json', '').replace('```', '').strip()
            queries = json.loads(cleaned_response)
            return [str(q) for q in queries[:3]]
        except Exception:
            logger.error(f"Failed to parse search queries: {response}")
            return [str(research_item)]

    def prioritize_unfulfilled_requirements(self, plan: Dict[str, List[str]], progress: Dict[str, bool], gathered_info: List[str] = None) -> List[tuple]:
        items = []
        def has_sufficient_depth(topic: str, info: List[str]) -> bool:
            if not info: return False
            mentions = sum(1 for text in info if topic.lower() in text.lower() and len(text) > 500)
            return mentions >= 2

        if not progress.get("core_concepts", False):
            for item in plan.get("core_concepts", []):
                if not has_sufficient_depth(item, gathered_info): items.append(("core_concepts", item))
        if not progress.get("key_questions", False):
            for item in plan.get("key_questions", []):
                if not has_sufficient_depth(item, gathered_info): items.append(("key_questions", item))
        if not progress.get("information_requirements", False):
            for item in plan.get("information_requirements", []):
                if not has_sufficient_depth(item, gathered_info): items.append(("information_requirements", item))
        return items

class ReportAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = """You are an expert technical writer and research analyst. Your task is to synthesize the provided research findings into a well-structured, comprehensive, and objective technical report, using Markdown for formatting. Adhere strictly to the facts and sources provided."""

    def generate_report(self, query: str, research_plan: Dict[str, List[str]], research_results: List[str], completion_stats: Dict[str, Any]) -> str:
        prompt = f"""
        **Objective:** Generate a comprehensive technical report on the topic: "{query}"

        **Original Research Plan:**
        ```json
        {json.dumps(research_plan, indent=2)}
        ```

        **Collected Research Context:**
        ---
        {chr(10).join(research_results)}
        ---

        **Report Generation Task:**
        Based *only* on the "Collected Research Context" provided above, write a detailed and well-structured technical report.

        **CRITICAL INSTRUCTIONS:**
        1.  **Synthesize, Don't Summarize:** Do not merely list facts from each source. Weave the information together into a coherent narrative. Connect ideas, compare and contrast findings, and build a logical argument.
        2.  **Structure Logically:** Start with a concise introduction. Group related information into sections with clear headings (e.g., `## Architecture`, `### Training Data`). The structure should emerge from the content, not a rigid template.
        3.  **Go Deep:** Prioritize technical depth. Explain mechanisms, architectures, and data points in detail. Avoid superficial statements.
        4.  **Acknowledge Sources (Implicitly):** While you don't need to add inline citations, your writing should clearly be based on the provided text.
        5.  **Conclude Powerfully:** End with a conclusion that summarizes the most important takeaways and implications.
        6.  **Omit Missing Information:** If the context for a planned section is weak or missing, DO NOT invent information. It is better to have a shorter, accurate report than a longer one with filler. Focus on what is well-supported by the context.
        """
        return self.generate(prompt, self.system_prompt)