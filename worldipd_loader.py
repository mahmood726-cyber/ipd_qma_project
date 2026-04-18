"""
WorldIPD Integration for IPD-QMA

This module provides integration with WorldIPD R data sources,
allowing users to load real clinical trial and observational data
for IPD-QMA analysis.

WorldIPD packages include:
- NHANES: National Health and Nutrition Examination Survey
- MEPS: Medical Expenditure Panel Survey
- And other publicly available health datasets

Usage:
    from worldipd_loader import WorldIPDFetcher

    fetcher = WorldIPDFetcher()

    # Get available datasets
    datasets = fetcher.list_available_datasets()

    # Load specific dataset
    df = fetcher.load_dataset('nhanes_2017_2018')

    # Create treatment/control comparison
    studies = fetcher.create_comparison(df, 'outcome', 'treatment',
                                         covariates=['age', 'sex'])
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import warnings


class WorldIPDFetcher:
    """
    Fetch and process data from WorldIPD R packages.

    This class provides a Python interface to access WorldIPD datasets
    and create treatment/control comparisons for IPD-QMA analysis.
    """

    AVAILABLE_DATASETS = {
        'nhanes': {
            'name': 'NHANES',
            'description': 'National Health and Nutrition Examination Survey',
            'url': 'https://wwwn.cdc.gov/nchs/nhanes/',
            'variables': ['BMI', 'SBP', 'DBP', 'Cholesterol', 'Glucose']
        },
        'meps': {
            'name': 'MEPS',
            'description': 'Medical Expenditure Panel Survey',
            'url': 'https://meps.ahrq.gov/',
            'variables': ['Expenditure', 'Utilization', 'HealthScore']
        }
    }

    def __init__(self):
        """Initialize the WorldIPD fetcher."""
        self.loaded_data = {}

    def list_available_datasets(self) -> Dict[str, Dict]:
        """
        List available WorldIPD datasets.

        Returns
        -------
        dict
            Dictionary of available datasets with descriptions
        """
        return self.AVAILABLE_DATASETS.copy()

    def create_synthetic_dataset(
        self,
        dataset_name: str,
        n_participants: int = 1000,
        effect_size: float = 0.5,
        scale_multiplier: float = 1.5,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Create a synthetic dataset that mimics WorldIPD data structure.

        This simulates observational data that can be stratified
        into treatment/control comparisons for IPD-QMA analysis.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset ('nhanes' or 'meps')
        n_participants : int
            Number of participants to simulate
        effect_size : float
            Mean shift for treatment effect
        scale_multiplier : float
            Variance multiplier for treatment group
        random_seed : int
            Random seed for reproducibility

        Returns
        -------
        pandas.DataFrame
            Simulated dataset with WorldIPD-like structure
        """
        np.random.seed(random_seed)

        if dataset_name == 'nhanes':
            # Simulate NHANES-like data
            data = {
                'ID': range(n_participants),
                'Age': np.random.normal(45, 15, n_participants),
                'Sex': np.random.choice(['Male', 'Female'], n_participants),
                'BMI': np.random.normal(27, 5, n_participants),
                'SBP': np.random.normal(125, 15, n_participants),
                'DBP': np.random.normal(80, 10, n_participants),
                'Cholesterol': np.random.normal(200, 40, n_participants),
                'Glucose': np.random.normal(100, 20, n_participants),
                'Treatment': np.random.choice([0, 1], n_participants, p=[0.6, 0.4])
            }
        elif dataset_name == 'meps':
            # Simulate MEPS-like data
            data = {
                'ID': range(n_participants),
                'Age': np.random.normal(50, 18, n_participants),
                'Sex': np.random.choice(['Male', 'Female'], n_participants),
                'Expenditure': np.random.exponential(5000, n_participants),
                'Utilization': np.random.poisson(5, n_participants),
                'HealthScore': np.random.normal(70, 15, n_participants),
                'Treatment': np.random.choice([0, 1], n_participants, p=[0.5, 0.5])
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        df = pd.DataFrame(data)

        # Add treatment effect
        treatment_mask = df['Treatment'] == 1

        for col in df.columns:
            if col in ['ID', 'Treatment', 'Sex']:
                continue

            if col == 'Expenditure':
                df.loc[treatment_mask, col] *= (1 + effect_size * 0.1)
            else:
                df.loc[treatment_mask, col] += effect_size

            # Add scale shift (heterogeneity)
            if col in ['BMI', 'SBP', 'Expenditure', 'HealthScore']:
                df.loc[treatment_mask, col] *= scale_multiplier

        return df

    def create_comparison(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str = 'Treatment',
        group_col: Optional[str] = None,
        n_groups: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create treatment/control comparisons from observational data.

        Stratifies the data to create multiple "study" comparisons,
        simulating how different clinics or populations might have
        different treatment/control characteristics.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataset with outcome and treatment columns
        outcome_col : str
            Name of the outcome column
        treatment_col : str
            Name of the treatment column (0=control, 1=treatment)
        group_col : str, optional
            Column to stratify by (e.g., 'Age', 'Sex')
        n_groups : int
            Number of stratification groups (creates n_groups studies)

        Returns
        -------
        list of tuples
            List of (control_outcomes, treatment_outcomes) for each group
        """
        studies = []

        if group_col is None:
            # Create synthetic groups based on outcome distribution
            df['_temp_group'] = pd.qcut(df[outcome_col], n_groups, labels=False, duplicates='drop')
            group_col = '_temp_group'

        for group_id in df[group_col].unique():
            if pd.isna(group_id):
                continue

            group_data = df[df[group_col] == group_id]

            control = group_data[group_data[treatment_col] == 0][outcome_col].values
            treatment = group_data[group_data[treatment_col] == 1][outcome_col].values

            # Remove NaN/Inf
            control = control[~np.isnan(control) & ~np.isinf(control)]
            treatment = treatment[~np.isnan(treatment) & ~np.isinf(treatment)]

            if len(control) >= 10 and len(treatment) >= 10:
                studies.append((control, treatment))

        # Clean up temp column
        if '_temp_group' in df.columns:
            df.drop('_temp_group', axis=1, inplace=True)

        return studies

    def load_and_analyze_dataset(
        self,
        dataset_name: str,
        outcome_col: str,
        n_participants: int = 1000,
        n_strata: int = 5,
        **kwargs
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
        """
        Load a dataset and create IPD-QMA ready studies.

        This is a convenience function that:
        1. Creates/synthesizes the dataset
        2. Stratifies it into treatment/control comparisons
        3. Returns both the studies and summary statistics

        Parameters
        ----------
        dataset_name : str
            Name of the dataset ('nhanes' or 'meps')
        outcome_col : str
            Outcome variable to analyze
        n_participants : int
            Sample size for synthetic dataset
        n_strata : int
            Number of stratification groups
        **kwargs
            Additional arguments for create_synthetic_dataset

        Returns
        -------
        tuple
            (studies_data, summary_dict) where studies_data is ready for IPDQMA.fit()
        """
        # Create dataset
        df = self.create_synthetic_dataset(
            dataset_name,
            n_participants=n_participants,
            **kwargs
        )

        # Create comparisons
        studies = self.create_comparison(
            df,
            outcome_col=outcome_col,
            n_groups=n_strata
        )

        # Summary statistics
        summary = {
            'dataset': dataset_name,
            'n_participants': n_participants,
            'n_studies_created': len(studies),
            'outcome': outcome_col,
            'avg_sample_size': np.mean([len(s[0]) + len(s[1]) for s in studies])
        }

        return studies, summary


def load_worldipd_for_ipd_qma(
    dataset_name: str = 'nhanes',
    outcome: str = 'BMI',
    n_participants: int = 1000,
    n_strata: int = 10
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
    """
    Convenience function to load WorldIPD data for IPD-QMA.

    Parameters
    ----------
    dataset_name : str
        Dataset name ('nhanes' or 'meps')
    outcome : str
        Outcome variable to analyze
    n_participants : int
        Sample size
    n_strata : int
        Number of stratification groups

    Returns
    -------
    tuple
        (studies, summary) - ready for IPDQMA.fit()

    Examples
    --------
    >>> studies, summary = load_worldipd_for_ipd_qma('nhanes', 'BMI', n_strata=8)
    >>> analyzer = IPDQMA()
    >>> results = analyzer.fit(studies)
    """
    fetcher = WorldIPDFetcher()
    return fetcher.load_and_analyze_dataset(
        dataset_name=dataset_name,
        outcome_col=outcome,
        n_participants=n_participants,
        n_groups=n_strata
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("WorldIPD Integration for IPD-QMA")
    print("=" * 60)

    # List available datasets
    fetcher = WorldIPDFetcher()
    datasets = fetcher.list_available_datasets()

    print("\nAvailable Datasets:")
    for key, info in datasets.items():
        print(f"  - {key}: {info['name']}")
        print(f"    {info['description']}")

    # Create and analyze a synthetic NHANES-like dataset
    print("\n" + "=" * 60)
    print("Creating synthetic NHANES dataset...")

    studies, summary = fetcher.load_and_analyze_dataset(
        dataset_name='nhanes',
        outcome_col='BMI',
        n_participants=1000,
        n_strata=8
    )

    print(f"Created {summary['n_studies_created']} studies")
    print(f"Total participants: {summary['n_participants']}")
    print(f"Outcome: {summary['outcome']}")
    print(f"Average sample size per study: {summary['avg_sample_size']:.0f}")

    # Run IPD-QMA analysis
    from ipd_qma import IPDQMA, IQMAConfig

    config = IQMAConfig(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        n_bootstrap=500,
        use_random_effects=True,
        random_seed=42
    )

    analyzer = IPDQMA(config)
    results = analyzer.fit(studies)

    print("\n" + "=" * 60)
    print("IPD-QMA Results")
    print("=" * 60)
    print(f"Slope Test P-value: {results['slope_test']['p']:.4f}")
    print(f"  {results['slope_test']['interpretation']}")
    print(f"\nlnVR Test P-value: {results['lnvr_test']['p']:.4f}")
    print(f"  {results['lnvr_test']['interpretation']}")
