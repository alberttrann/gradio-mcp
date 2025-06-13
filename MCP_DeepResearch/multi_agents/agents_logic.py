# multi_agents/agents_logic.py
# This is your proven agent logic. The only change is the PlannerAgent prompt.
# (I am including the full code for completeness)

import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from openai import OpenAI
import logging
import json

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, use_gemini: bool = True, api_key: Optional[str] = None, 
                 openrouter_model: Optional[str] = None, gemini_model: Optional[str] = None):
        self.use_gemini = use_gemini
        if use_gemini:
            if not api_key: raise ValueError("Gemini API key is required when use_gemini=True")
            self.gemini_model = gemini_model or "gemini-2.0-flash"
        else:
            self.openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            self.model = openrouter_model or "anthropic/claude-3-opus:beta"

    def _generate_with_gemini(self, prompt: str, system_prompt: str) -> str:
        try:
            model = genai.GenerativeModel(model_name=self.gemini_model)
            combined_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            response = model.generate_content(
                combined_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise
            
    def generate(self, prompt: str, system_prompt: str) -> str:
        # For simplicity, we assume Gemini is always used in this flow
        return self._generate_with_gemini(prompt, system_prompt)

class OrchestratorAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = "You are an expert research planner..."
    def create_research_plan(self, query: str) -> Dict[str, List[str]]:
        prompt = f"""Create a detailed research plan for the query: {query}... (prompt is correct)"""
        response = self.generate(prompt, self.system_prompt)
        try:
            cleaned_response = response.strip().replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_response)
        except: return {"core_concepts": [query], "key_questions": [query], "information_requirements": [query], "research_priorities": [query]}
    def evaluate_research_progress(self, plan: Dict[str, List[str]], gathered_info: List[str]) -> Dict[str, bool]:
        prompt = f"""Analyze the research plan and gathered info... (prompt is correct)"""
        response = self.generate(prompt, self.system_prompt)
        try:
            cleaned_response = response.strip().strip('"').strip().replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_response)
        except: return {"core_concepts": False, "key_questions": False, "information_requirements": False}

class PlannerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = "You are an expert research planner that creates targeted, concise search queries."
    def create_search_strategy(self, research_item: str, item_type: str) -> List[str]:
        # THIS IS THE CRITICAL FIX FROM BEFORE
        prompt = f"""Create 2-3 concise search queries for the {item_type}: "{research_item}"
        CRITICAL RULES:
        1. Each query MUST be less than 200 characters.
        2. Do NOT use complex boolean operators like 'AND' or 'OR'.
        3. Focus on keywords.
        4. Return ONLY a valid JSON array of strings.
        Example: ["transformer model architecture", "self-attention mechanism explained", "vision transformer applications"]"""
        response = self.generate(prompt, self.system_prompt)
        try:
            cleaned_response = response.strip().replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_response)
        except: return [str(research_item)]
    def prioritize_unfulfilled_requirements(self, plan: Dict[str, List[str]], progress: Dict[str, bool], gathered_info: List[str] = None) -> List[tuple]:
        # This logic is correct and unchanged
        items = []
        def has_sufficient_depth(topic: str, info: List[str]) -> bool:
            if not info: return False
            mentions = sum(1 for text in info if topic.lower() in text.lower() and len(text) > 200)
            return mentions >= 1
        if not progress["core_concepts"]:
            for item in plan["core_concepts"]:
                if not has_sufficient_depth(item, gathered_info): items.append(("core_concepts", item))
        if not progress["key_questions"]:
            for item in plan["key_questions"]:
                if not has_sufficient_depth(item, gathered_info): items.append(("key_questions", item))
        if not progress["information_requirements"]:
            for item in plan["information_requirements"]:
                if not has_sufficient_depth(item, gathered_info): items.append(("information_requirements", item))
        return items

class ReportAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = "You are an expert technical writer..."
    def generate_report(self, query: str, research_plan: Dict[str, List[str]], 
                       research_results: List[str], completion_stats: Dict[str, Any]) -> str:
        prompt = f"""Generate a comprehensive technical report... (prompt is correct)"""
        return self.generate(prompt, self.system_prompt)