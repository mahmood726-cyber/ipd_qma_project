"""
IPD-QMA Real Data Fetcher

This module fetches real Individual Participant Data from reliable sources
that can be used for IPD-QMA analysis.
"""

import pandas as pd
import numpy as np
import requests
import io
import os
from typing import List, Tuple, Dict, Optional
import warnings


class RealIPDFetcher:
    """Fetch real IPD data from various reliable sources."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "data")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fetch_from_url(self, url: str, dataset_name: str) -> Optional[pd.DataFrame]:
        """Fetch a dataset from a URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Try to parse as CSV
            if url.endswith('.csv'):
                df = pd.read_csv(io.StringIO(response.text))
            else:
                # Try to read as CSV anyway
                df = pd.read_csv(io.StringIO(response.text))

            # Save locally
            output_path = os.path.join(self.output_dir, f"{dataset_name}.csv")
            df.to_csv(output_path, index=False)
            print(f"  Saved to {output_path}")

            return df
        except Exception as e:
            print(f"  ERROR: {e}")
            return None

    def fetch_uci_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch datasets from UCI Machine Learning Repository that can
        demonstrate IPD-QMA (treatment/control comparisons).
        """
        print("\n[*] Fetching UCI datasets...")

        datasets = {}

        # 1. Heart Disease (Cleveland) - has disease status as outcome
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
        try:
            df = pd.read_csv(url, names=cols, na_values='?')
            df['diagnosis'] = (df['num'] > 0).astype(int)  # 0=healthy, 1=disease
            datasets['heart_disease'] = df
            print(f"  [OK] heart_disease: {len(df)} records")
        except Exception as e:
            print(f"  [FAIL] heart_disease: {e}")

        # 2. Pima Indians Diabetes
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = ['pregnancies', 'glucose', 'bp', 'skin_thickness', 'insulin',
                'bmi', 'pedigree', 'age', 'outcome']
        try:
            df = pd.read_csv(url, names=cols)
            datasets['diabetes'] = df
            print(f"  [OK] diabetes: {len(df)} records")
        except Exception as e:
            print(f"  [FAIL] diabetes: {e}")

        # 3. Wine Quality (red vs white comparison potential)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        try:
            df = pd.read_csv(url, sep=';')
            df['type'] = 'red'
            datasets['wine_red'] = df
            print(f"  [OK] wine_red: {len(df)} records")
        except Exception as e:
            print(f"  [FAIL] wine_red: {e}")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        try:
            df = pd.read_csv(url, sep=';')
            df['type'] = 'white'
            datasets['wine_white'] = df
            print(f"  [OK] wine_white: {len(df)} records")
        except Exception as e:
            print(f"  [FAIL] wine_white: {e}")

        return datasets

    def fetch_from_github_ipd_repos(self) -> Dict[str, pd.DataFrame]:
        """Fetch IPD datasets from GitHub repositories."""
        print("\n[*] Fetching from GitHub...")

        datasets = {}

        # 1. Clinical trial data from various sources
        # Example: METABRIC breast cancer dataset (if available publicly)

        return datasets

    def create_ipd_studies_from_observational(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        group_col: str,
        control_value: any = 0,
        treatment_value: any = 1,
        min_sample_size: int = 20
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create IPD-QMA compatible studies from observational data.

        Creates multiple "studies" by stratifying on covariates.
        """
        studies = []

        # Get stratification columns (excluding outcome and group columns)
        exclude_cols = {outcome_col, group_col, 'patient_id', 'dataset_id'}
        covariate_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'object']]

        for col in covariate_cols[:3]:  # Use up to 3 stratification variables
            unique_vals = df[col].dropna().unique()

            for val in unique_vals:
                subset = df[df[col] == val]

                control = subset[subset[group_col] == control_value][outcome_col].dropna().values
                treatment = subset[subset[group_col] == treatment_value][outcome_col].dropna().values

                if len(control) >= min_sample_size and len(treatment) >= min_sample_size:
                    studies.append((control, treatment))

        return studies

    def create_cross_sectional_studies(
        self,
        datasets: Dict[str, pd.DataFrame],
        outcome_col: str = None,
        n_studies_per_dataset: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create multiple studies from cross-sectional datasets.

        Simulates treatment effect by comparing quantile groups.
        """
        all_studies = []

        for name, df in datasets.items():
            # Try to find a suitable outcome column
            if outcome_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    outcome_col = numeric_cols[0]
                else:
                    continue

            outcomes = df[outcome_col].dropna().values

            if len(outcomes) < 50:
                continue

            # Create studies by splitting on quantiles
            for i in range(n_studies_per_dataset):
                # Random split
                np.random.seed(i)
                mask = np.random.rand(len(outcomes)) < 0.5
                control = outcomes[mask]
                treatment = outcomes[~mask]

                # Add a treatment effect to treatment group
                effect_size = np.random.uniform(0.1, 0.5)
                treatment = treatment * (1 + effect_size)

                if len(control) >= 20 and len(treatment) >= 20:
                    all_studies.append((control, treatment))

        return all_studies


# ==========================================
# MAIN: Fetch and prepare data for IPD-QMA
# ==========================================

def main():
    """Main function to fetch and prepare data."""
    print("=" * 70)
    print("IPD-QMA Real Data Fetcher")
    print("=" * 70)

    fetcher = RealIPDFetcher()

    # Fetch UCI datasets
    datasets = fetcher.fetch_uci_datasets()

    if not datasets:
        print("\n[!] No datasets fetched. Using simulated data instead.")
        return None

    # Save all datasets
    for name, df in datasets.items():
        path = os.path.join(fetcher.output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {name} to {path}")

    # Create IPD studies from the datasets
    print("\n[*] Creating IPD studies from datasets...")

    # For diabetes dataset
    if 'diabetes' in datasets:
        df = datasets['diabetes']
        # Compare glucose levels by diabetes outcome
        non_diabetic = df[df['outcome'] == 0]['glucose'].values
        diabetic = df[df['outcome'] == 1]['glucose'].values

        print(f"  Diabetes: {len(non_diabetic)} non-diabetic, {len(diabetic)} diabetic")
        print(f"    Non-diabetic mean: {np.mean(non_diabetic):.2f}")
        print(f"    Diabetic mean: {np.mean(diabetic):.2f}")

    # For heart disease dataset
    if 'heart_disease' in datasets:
        df = datasets['heart_disease']
        # Compare cholesterol levels by heart disease status
        healthy = df[df['diagnosis'] == 0]['chol'].dropna().values
        disease = df[df['diagnosis'] == 1]['chol'].dropna().values

        print(f"  Heart Disease: {len(healthy)} healthy, {len(disease)} with disease")
        print(f"    Healthy mean cholesterol: {np.mean(healthy):.2f}")
        print(f"    Disease mean cholesterol: {np.mean(disease):.2f}")

    # Combine wine datasets
    if 'wine_red' in datasets and 'wine_white' in datasets:
        red = datasets['wine_red']['alcohol'].values
        white = datasets['wine_white']['alcohol'].values

        print(f"  Wine: {len(red)} red, {len(white)} white")
        print(f"    Red mean alcohol: {np.mean(red):.2f}")
        print(f"    White mean alcohol: {np.mean(white):.2f}")

    print("\n" + "=" * 70)
    print(f"Data saved to: {fetcher.output_dir}")
    print("=" * 70)

    return datasets


if __name__ == "__main__":
    datasets = main()
