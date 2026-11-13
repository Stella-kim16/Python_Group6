import matplotlib.pyplot as plt
from typing import List
from subsystem3.optimisation_result import OptimisationResult
from subsystem3.exceptions_subsystem3 import VisualisationError

class ResultsVisualiser:
    """Renders visual performance charts for the energy management dashboard."""
    
    def __init__(self, title: str = "Energy Optimisation Report", library: str = "matplotlib"):
        self._title = title
        self._library = library
        self.setup_plot_style()

    def setup_plot_style(self) -> None:
        """Configure matplotlib style parameters."""
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Getters & Setters for Encapsulation
    def get_title(self) -> str:
        return self._title

    def set_title(self, title: str) -> None:
        self._title = title

    def get_library(self) -> str:
        return self._library

    def set_library(self, library: str) -> None:
        self._library = library

    def plot_energy_comparison(self, results: List[OptimisationResult]) -> plt.Figure:
        """Plots energy consumption comparison across scenarios using getters."""
        if not results:
            raise VisualisationError("Cannot plot energy comparison for an empty results list.")
        descriptions = [r.get_description() for r in results]
        energies = [r.get_energy_kwh() for r in results]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(descriptions, energies, color=['skyblue' if not r.is_recommended() else 'forestgreen' for r in results])
        ax.set_xlabel("Total Energy Consumption (kWh)")
        ax.set_title("Scenario Energy Comparison")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + 5, bar.get_y() + bar.get_height()/2.0, f"{xval:.1f} kWh", ha='left', va='center')
            
        fig.tight_layout()
        return fig

    def plot_cost_comparison(self, results: List[OptimisationResult]) -> plt.Figure:
        """Plots cost comparison across scenarios using getters."""
        if not results:
            raise VisualisationError("Cannot plot cost comparison for an empty results list.")
        descriptions = [r.get_description() for r in results]
        costs = [r.get_cost_eur() for r in results]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(descriptions, costs, color=['orange' if not r.is_recommended() else 'darkgreen' for r in results])
        ax.set_xlabel("Total Cost (EUR)")
        ax.set_title("Scenario Operating Cost Comparison")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + 1, bar.get_y() + bar.get_height()/2.0, f"€{xval:.2f}", ha='left', va='center')
            
        fig.tight_layout()
        return fig