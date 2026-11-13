import os
import json
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any, List


SUBSYSTEM2_JSON = Path(__file__).resolve().parent.parent / "data" / "output" / "subsystem2_results.json"


class AIAdvisor:
    """Uses KI:connect NRW API to generate natural language insights based on Subsystem 2 and 3 data."""
    
    def __init__(self, api_key: str = None):
        # Initialize the OpenAI client pointing to KI:connect NRW backend
        # It's best practice to load the API key from an environment variable
        self._api_key = api_key or os.getenv("KICONNECT_API_KEY", "YOUR_API_KEY")
        self._client = OpenAI(
            base_url="https://chat.kiconnect.nrw/api/v1",
            api_key=self._api_key
        )

    def generate_optimization_insights(self, optimal_result: Dict[str, Any], subsystem2_data_path: str | Path = SUBSYSTEM2_JSON) -> str:
        """Reads Subsystem 2 JSON data and optimization results, then asks KI:connect to write an operator summary."""
        
        # 1. Load Subsystem 2 JSON data from data/processed/
        sub2_data = {}
        if os.path.exists(subsystem2_data_path):
            try:
                with open(subsystem2_data_path, 'r', encoding='utf-8') as f:
                    sub2_data = json.load(f)
            except Exception as e:
                sub2_data = {"error": f"Could not read file: {e}"}

        # 2. Construct prompt payload combining Subsystem 2 telemetry & Subsystem 3 results
        prompt = f"""
        You are an expert energy management AI assistant. 
        Here is the telemetry and forecasted data from Subsystem 2:
        {json.dumps(sub2_data, indent=2)}

        Here is the selected optimal mitigation strategy evaluated by Subsystem 3:
        {json.dumps(optimal_result, indent=2)}

        Provide a concise, professional 3-bullet-point summary for building operators explaining:
        1. Why this optimal scenario was selected.
        2. The projected impact on energy efficiency and carbon emissions.
        3. Actionable next steps for the facility manager.
        """

        try:
            # 3. Call KI:connect NRW API chat completion
            response = self._client.chat.completions.create(
                model="mistralai-mistral-small-4", # Or your preferred model available on KI:connect
                messages=[
                    {"role": "system", "content": "You are a smart energy management assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Advisory Service temporarily unavailable: {str(e)}"