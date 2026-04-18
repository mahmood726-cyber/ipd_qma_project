"""
IPD-QMA Data Loader for WorldIPD Datasets

This module helps load and prepare data from the WorldIPD R package structure
for use with IPD-QMA analysis.
"""

import pandas as pd
import numpy as np
import os
import subprocess
from typing import List, Tuple, Dict, Optional
import warnings


class WorldIPDLoader:
    """
    Load and prepare IPD data from WorldIPD datasets.

    WorldIPD contains observational datasets (NHANES, MEPS, NSDUH).
    For IPD-QMA, we need to create treatment/control comparisons.
    """

    def __init__(self, worldipd_path: str = r"C:\Users\user\OneDrive - NHS\Documents\WorldIPD"):
        self.worldipd_path = worldipd_path
        self.private_path = worldipd_path.replace("WorldIPD", "WorldIPD-private")
        self.extdata = os.path.join(worldipd_path, "inst", "extdata")
        self.registry = os.path.join(worldipd_path, "inst", "registry", "registry.csv")

    def list_available_datasets(self) -> pd.DataFrame:
        """List all registered datasets."""
        return pd.read_csv(self.registry)

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a specific dataset by ID."""
        datafile = os.path.join(self.extdata, f"{dataset_id}.csv")
        if os.path.exists(datafile):
            return pd.read_csv(datafile)
        else:
            raise FileNotFoundError(f"Dataset {dataset_id} not found locally. Fetch with R first.")

    def fetch_with_r(self, script_name: str) -> Dict[str, str]:
        """
        Fetch data using WorldIPD R scripts.

        Parameters
        ----------
        script_name : str
            Name of the R script (e.g., 'fetch_nhanes.R')

        Returns
        -------
        dict
            Status and output from R
        """
        script_path = os.path.join(self.worldipd_path, "data-raw", "fetchers", script_name)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"R script not found: {script_path}")

        try:
            result = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except FileNotFoundError:
            return {
                "success": False,
                "stdout": "",
                "stderr": "R not found. Please install R or use it directly."
            }


def create_treatment_control_from_observational(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    control_value: any = 0,
    treatment_value: any = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create treatment/control groups from an observational dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The observational dataset
    outcome_col : str
        Name of the outcome column
    group_col : str
        Column defining treatment/control groups
    control_value : any
        Value indicating control group
    treatment_value : any
        Value indicating treatment group

    Returns
    -------
    tuple
        (control_outcomes, treatment_outcomes)
    """
    control = df[df[group_col] == control_value][outcome_col].dropna().values
    treatment = df[df[group_col] == treatment_value][outcome_col].dropna().values

    if len(control) == 0 or len(treatment) == 0:
        raise ValueError(f"No data found for control={control_value} or treatment={treatment_value}")

    return control, treatment


def create_quantile_groups(
    df: pd.DataFrame,
    outcome_col: str,
    n_groups: int = 2,
    group_labels: Optional[List[str]] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create treatment/control comparisons by splitting on outcome quantiles.

    This simulates treatment effects by comparing different quantile groups
    (useful for demonstration).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with outcome variable
    outcome_col : str
        Name of the outcome column
    n_groups : int
        Number of quantile groups to create
    group_labels : list, optional
        Labels for groups

    Returns
    -------
    list of tuples
        List of (control, treatment) comparisons
    """
    outcomes = df[outcome_col].dropna().values
    quantile_cutoffs = np.linspace(0, 1, n_groups + 1)

    studies = []
    for i in range(n_groups - 1):
        lower = np.percentile(outcomes, quantile_cutoffs[i] * 100)
        upper = np.percentile(outcomes, quantile_cutoffs[i + 2] * 100)

        control = outcomes[(outcomes >= lower) & (outcomes < np.percentile(outcomes, quantile_cutoffs[i + 1] * 100))]
        treatment = outcomes[(outcomes >= np.percentile(outcomes, quantile_cutoffs[i + 1] * 100)) & (outcomes <= upper)]

        if len(control) > 10 and len(treatment) > 10:
            studies.append((control, treatment))

    return studies


# ==========================================
# EXAMPLE: Simulated IPD from Public Sources
# ==========================================

def generate_example_ipd_from_nhanes_structure(
    n_studies: int = 10,
    n_per_group: int = 100,
    effect_heterogeneity: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate example IPD that mimics NHANES-like data structure.

    This creates simulated data with realistic properties for testing IPD-QMA.

    Parameters
    ----------
    n_studies : int
        Number of studies to simulate
    n_per_group : int
        Sample size per group
    effect_heterogeneity : bool
        Whether to include heterogeneous treatment effects

    Returns
    -------
    list of tuples
        List of (control, treatment) arrays
    """
    studies = []
    np.random.seed(42)

    for i in range(n_studies):
        # Base parameters (vary across studies)
        base_mean = np.random.normal(0, 0.5)
        base_sd = np.random.uniform(0.8, 1.2)

        # Control group: skewed distribution (like health outcomes)
        control = np.random.exponential(base_sd, n_per_group) + base_mean - 1

        # Treatment group
        if effect_heterogeneity:
            # Variance shift + location shift
            variance_mult = np.random.uniform(2, 4)
            treatment = (np.random.exponential(base_sd, n_per_group) + base_mean - 1) * variance_mult
        else:
            # Simple location shift
            treatment = control + np.random.normal(0.3, 0.1, n_per_group)

        studies.append((control, treatment))

    return studies


def create_study_from_binary_treatment(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: Optional[List[str]] = None,
    min_sample_size: int = 20
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create multiple study comparisons from a dataset with binary treatment.

    Can stratify by covariates to create multiple "studies".

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with binary treatment indicator
    outcome_col : str
        Outcome variable name
    treatment_col : str
        Binary treatment column (0=control, 1=treatment)
    covariate_cols : list, optional
        Columns to stratify by (creates one study per stratum)
    min_sample_size : int
        Minimum sample size per group

    Returns
    -------
    list of tuples
        List of (control, treatment) for each stratum
    """
    studies = []

    if covariate_cols is None:
        # Single comparison
        control = df[df[treatment_col] == 0][outcome_col].dropna().values
        treatment = df[df[treatment_col] == 1][outcome_col].dropna().values

        if len(control) >= min_sample_size and len(treatment) >= min_sample_size:
            studies.append((control, treatment))
    else:
        # Stratified comparisons
        for col in covariate_cols:
            for stratum_val in df[col].dropna().unique():
                subset = df[df[col] == stratum_val]

                control = subset[subset[treatment_col] == 0][outcome_col].dropna().values
                treatment = subset[subset[treatment_col] == 1][outcome_col].dropna().values

                if len(control) >= min_sample_size and len(treatment) >= min_sample_size:
                    studies.append((control, treatment))

    return studies


# ==========================================
# QUICK START DEMO
# ==========================================

def demo_with_simulated_data():
    """Demonstrate IPD-QMA with simulated data."""
    from ipd_qma import IPDQMA, IQMAConfig

    print("=" * 70)
    print("IPD-QMA Demo: Simulated IPD Data")
    print("=" * 70)

    # Generate simulated data (mimics real IPD structure)
    studies = generate_example_ipd_from_nhanes_structure(
        n_studies=15,
        n_per_group=100,
        effect_heterogeneity=True
    )

    print(f"\nGenerated {len(studies)} studies with heterogeneous effects")
    print(f"Sample sizes: {[(len(c), len(t)) for c, t in studies[:3]]}...")

    # Run IPD-QMA
    config = IQMAConfig(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        n_bootstrap=500,
        use_random_effects=True
    )

    analyzer = IPDQMA(config)
    results = analyzer.fit(studies)

    # Display results
    analyzer.summary()
    analyzer.plot()
    plt.show()

    return analyzer, results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    demo_with_simulated_data()
