from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from subsystem3.optimisation_engine import OptimisationEngine
from subsystem3.ai_advisor import AIAdvisor

app = FastAPI(
    title="Smart Energy Management System - Subsystem 3",
    description="Optimization pipeline integrating Subsystem 2 data and KI:connect NRW AI insights."
)

@app.post("/subsystem3/optimize", summary="Run Optimization and Generate AI Insights")
def run_optimization_pipeline() -> Dict[str, Any]:
    """
    1. Loads Subsystem 2 data from data/processed/enriched_energy_data.json.
    2. Runs Subsystem 3 scenario evaluation and selects the best strategy.
    3. Calls KI:connect NRW API to generate a professional operator summary.
    """
    json_path = "data/processed/enriched_energy_data.json"
    
    try:
        # Step 1: Initialize Engine (automatically reads Subsystem 2 JSON)
        engine = OptimisationEngine(subsystem2_json_path=json_path)
        
        # Step 2: Evaluate scenarios
        results = engine.evaluate_scenarios()
        
        # Step 3: Select best scenario
        best_scenario = engine.select_best_scenario(results)
        
        # Step 4: Generate AI insights via KI:connect NRW
        advisor = AIAdvisor()
        ai_insights = advisor.generate_optimization_insights(
            subsystem2_data_path=json_path,
            optimal_result=best_scenario.summary()
        )
        
        return {
            "status": "success",
            "recommended_strategy": best_scenario.summary(),
            "ai_operator_insights": ai_insights,
            "all_scenarios": [r.summary() for r in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))