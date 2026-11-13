import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from subsystem2.AI_model import AImodel
from subsystem2.Demandforecast import DemandForecaster
from subsystem2.AI_savingadviser import SavingsAdvisor
from subsystem2.AI_sustainability import SustainabilityScorer
from subsystem2.Data_processor import load_clean_data

# Import Subsystem 3 classes
from subsystem3.optimisation_engine import OptimisationEngine
from subsystem3.results_visualiser import ResultsVisualiser

st.set_page_config(page_title="Smart Energy Management Dashboard", page_icon="⚡", layout="wide")

st.title("AI-Powered Smart Energy Management Dashboard")

# Load data and model with caching
@st.cache_resource
def load_system():
    PROJECT_ROOT = Path(__file__).resolve().parent
    PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "enriched_energy_data.csv"
    raw_df = pd.read_csv(PROCESSED_CSV)
    raw_df["Timestamp"] = pd.to_datetime(raw_df["Timestamp"])
    clean_df = load_clean_data(PROCESSED_CSV)
    model = AImodel(n_estimators=200, random_state=42)
    model.train(clean_df)
    return raw_df, model

raw_df, model = load_system()

# Sidebar Navigation & Filters
st.sidebar.header("Navigation & Filters")
app_mode = st.sidebar.radio(
    "Go to:", 
    [
        " Building Overview & Forecasts", 
        " Saving Advisor", 
        " Sustainability Scoring", 
        " Optimisation & Visualisation", 
        " Audit Reports & Export"
    ]
)

building_id = st.sidebar.selectbox("Select Building", sorted(raw_df["building_id"].unique()))

# Filter data for selected building
bld_raw = raw_df[raw_df["building_id"] == building_id]

if app_mode == " Building Overview & Forecasts":
    st.header(f" Operational Analytics: {building_id}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(bld_raw))
    col2.metric("Total Energy (kWh)", f"{bld_raw['EnergyConsumption'].sum():,.1f}")
    col3.metric("Total Cost (EUR)", f"€{bld_raw['energy_cost_eur'].sum():,.2f}")
    col4.metric("Carbon Footprint (kg)", f"{bld_raw['carbon_footprint_kg'].sum():,.1f} kg")

    st.subheader("Recent Energy Consumption Trend")
    chart_data = bld_raw.set_index("Timestamp")["EnergyConsumption"]
    st.line_chart(chart_data)

    st.subheader("Next 24-Hour AI Demand Forecast")
    forecaster = DemandForecaster(model, raw_df)
    forecast = forecaster.forecast_building(building_id)
    st.line_chart(forecast.hourly["predicted_kWh"])
    
    fc1, fc2 = st.columns(2)
    fc1.metric("Forecasted 24h kWh", f"{forecast.total_kwh:,.2f} kWh")
    fc2.metric("Forecasted 24h Cost", f"€{forecast.total_cost_eur:,.2f}")

elif app_mode == " Saving Advisor":
    st.header(f" Energy Saving Opportunities: {building_id}")
    advisor = SavingsAdvisor(model, raw_df)
    recs = advisor.recommend(building_id)
    
    if recs:
        for rec in recs:
            st.warning(f"**{rec.title}**\n\n{rec.message}\n\n*Potential Savings:* **~€{rec.saved_eur_per_month:.2f} / month** ({rec.saved_pct}% reduction)")
    else:
        st.success(" Excellent! No active waste conditions detected for this building.")

elif app_mode == " Sustainability Scoring":
    st.header(f" Sustainability Performance: {building_id}")
    scorer = SustainabilityScorer(model, raw_df)
    score_res = scorer.score_building(building_id)
    
    sc1, sc2 = st.columns(2)
    sc1.metric("Total Sustainability Score", f"{score_res.total_score} / 100", f"Type: {score_res.building_type}")
    sc2.metric("Efficiency Score", f"{score_res.efficiency_score} / 100")
    
    st.subheader("Sub-Score Breakdown")
    sub_df = pd.DataFrame(list(score_res.sub_scores.items()), columns=["Category", "Score (0-100)"])
    st.bar_chart(sub_df.set_index("Category"))
    
    st.subheader("Audit Metrics Details")
    st.json(score_res.details)

elif app_mode == " Optimisation & Visualisation":
    st.header(f" Subsystem 3: Operational Optimisation & Visualisation ({building_id})")
    st.markdown("Simulate operational scenarios, evaluate energy/cost/carbon trade-offs, render performance charts, and generate real-time AI operator insights via KI:connect NRW.")

    #1.Evaluate scenarios for the selected building
    engine = OptimisationEngine(building_id=building_id)
    results = engine.evaluate_scenarios()

    best_scenario = engine.select_best_scenario(results)

    st.success(f" **Recommended Optimal Strategy:** {best_scenario.get_description()} (Score: {best_scenario.get_score()}/100)")

    # 2. Display AI Operator Insights from KI:connect NRW API
    st.subheader(" KI:connect NRW AI Operator Advisor Insights")
    with st.spinner("Querying KI:connect NRW LLM for operational recommendations..."):
        try:
            from subsystem3.ai_advisor import AIAdvisor
            advisor = AIAdvisor()
            
            # Uses the Subsystem 2 results JSON by default
            ai_insights = advisor.generate_optimization_insights(
                subsystem2_data_path="data/output/subsystem2_results.json",
                optimal_result=best_scenario.summary()
            )
            # Display the AI insights in a clean Streamlit container
            st.info(ai_insights)
        except Exception as e:
            st.warning(f"Could not load AI Advisor insights: {str(e)}")

    # 3. Display summary table of evaluated scenarios
    st.subheader(" Evaluated Optimisation Scenarios")
    scenario_summary_df = pd.DataFrame([r.summary() for r in results])
    st.dataframe(scenario_summary_df, use_container_width=True)

    # 4. Render Matplotlib Visualisations using ResultsVisualiser class
    st.subheader(" Subsystem 3 Matplotlib Visualisation Dashboard")
    visualiser = ResultsVisualiser(title=f"Optimisation Analysis for {building_id}")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("**Energy Comparison Across Scenarios**")
        fig_energy = visualiser.plot_energy_comparison(results)
        st.pyplot(fig_energy)
        
    with col_v2:
        st.markdown("**Operating Cost Comparison Across Scenarios**")
        fig_cost = visualiser.plot_cost_comparison(results)
        st.pyplot(fig_cost)

elif app_mode == " Audit Reports & Export":
    st.header(" Comprehensive Audit Reports")
    st.markdown("Generate and download certified audit reports containing historical telemetry, anomaly tracking, and AI predictive forecasts.")
    
    report_type = st.selectbox("Select Audit Report Type", ["Historical Telemetry & Anomalies", "AI 24-Hour Forecast & Financial Audit", "Complete Building Portfolio Summary"])
    
    if report_type == "Historical Telemetry & Anomalies":
        st.subheader(f"Historical Audit: {building_id}")
        audit_df = bld_raw[["Timestamp", "building_id", "building_type", "SquareFootage", "EnergyConsumption", "Occupancy", "HVACUsage", "LightingUsage", "energy_cost_eur", "carbon_footprint_kg", "is_anomaly"]].copy()
        
        st.dataframe(audit_df, use_container_width=True)
        
        # CSV Export
        csv_data = audit_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Download Historical Audit CSV",
            data=csv_data,
            file_name=f"historical_audit_{building_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    elif report_type == "AI 24-Hour Forecast & Financial Audit":
        st.subheader(f"Prediction & Financial Audit: {building_id}")
        forecaster = DemandForecaster(model, raw_df)
        forecast = forecaster.forecast_building(building_id)
        
        forecast_audit_df = forecast.hourly.reset_index()
        forecast_audit_df.rename(columns={"index": "Timestamp"}, inplace=True)
        
        st.dataframe(forecast_audit_df, use_container_width=True)
        
        fc_csv = forecast_audit_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Download Prediction Audit CSV",
            data=fc_csv,
            file_name=f"prediction_audit_{building_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    elif report_type == "Complete Building Portfolio Summary":
        st.subheader("Portfolio-Wide Sustainability & Savings Audit")
        scorer = SustainabilityScorer(model, raw_df)
        all_scores = scorer.score_all()
        
        summary_rows = []
        for bld, res in all_scores.items():
            summary_rows.append({
                "Building ID": bld,
                "Type": res.building_type,
                "Total Score": res.total_score,
                "Efficiency Score": res.efficiency_score,
                "Waste Score": res.sub_scores.get("waste"),
                "Renewable Score": res.sub_scores.get("renewable"),
                "Peak Score": res.sub_scores.get("peak"),
            })
        
        portfolio_df = pd.DataFrame(summary_rows)
        st.dataframe(portfolio_df, use_container_width=True)
        
        port_csv = portfolio_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Download Portfolio Audit Summary CSV",
            data=port_csv,
            file_name=f"portfolio_audit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )