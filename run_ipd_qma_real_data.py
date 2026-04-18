"""
IPD-QMA Analysis with Real UCI Datasets

This script demonstrates IPD-QMA using real datasets from UCI Machine Learning Repository.
We create treatment/control comparisons from observational data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ipd_qma import IPDQMA, IQMAConfig


def load_uci_data(data_dir: str = "data"):
    """Load the fetched UCI datasets."""
    datasets = {}
    data_path = os.path.join(os.path.dirname(__file__), data_dir)

    files = {
        'diabetes': 'diabetes.csv',
        'heart_disease': 'heart_disease.csv',
        'wine_red': 'wine_red.csv',
        'wine_white': 'wine_white.csv'
    }

    for name, filename in files.items():
        path = os.path.join(data_path, filename)
        if os.path.exists(path):
            datasets[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(datasets[name])} records")

    return datasets


def create_studies_from_datasets(datasets: dict) -> list:
    """
    Create IPD-QMA compatible studies from the UCI datasets.

    Each "study" is a treatment/control comparison.
    """
    studies = []

    # 1. Diabetes Dataset - Compare glucose levels
    if 'diabetes' in datasets:
        df = datasets['diabetes']

        # Non-diabetic vs Diabetic - Glucose comparison
        control = df[df['outcome'] == 0]['glucose'].values
        treatment = df[df['outcome'] == 1]['glucose'].values

        # Remove outliers and clean data
        control = control[(control > 0) & (control < 300)]
        treatment = treatment[(treatment > 0) & (treatment < 300)]

        if len(control) > 20 and len(treatment) > 20:
            studies.append({
                'name': 'Diabetes: Glucose',
                'control': control,
                'treatment': treatment,
                'description': 'Non-diabetic vs Diabetic (Glucose levels)'
            })

        # BMI comparison
        control = df[df['outcome'] == 0]['bmi'].values
        treatment = df[df['outcome'] == 1]['bmi'].values

        control = control[(control > 10) & (control < 60)]
        treatment = treatment[(treatment > 10) & (treatment < 60)]

        if len(control) > 20 and len(treatment) > 20:
            studies.append({
                'name': 'Diabetes: BMI',
                'control': control,
                'treatment': treatment,
                'description': 'Non-diabetic vs Diabetic (BMI)'
            })

        # Blood Pressure comparison
        control = df[df['outcome'] == 0]['bp'].values
        treatment = df[df['outcome'] == 1]['bp'].values

        control = control[(control > 40) & (control < 150)]
        treatment = treatment[(treatment > 40) & (treatment < 150)]

        if len(control) > 20 and len(treatment) > 20:
            studies.append({
                'name': 'Diabetes: Blood Pressure',
                'control': control,
                'treatment': treatment,
                'description': 'Non-diabetic vs Diabetic (Blood Pressure)'
            })

    # 2. Heart Disease Dataset
    if 'heart_disease' in datasets:
        df = datasets['heart_disease']

        # Cholesterol comparison
        control = df[df['diagnosis'] == 0]['chol'].dropna().values
        treatment = df[df['diagnosis'] == 1]['chol'].dropna().values

        control = control[(control > 100) & (control < 400)]
        treatment = treatment[(treatment > 100) & (treatment < 400)]

        if len(control) > 20 and len(treatment) > 20:
            studies.append({
                'name': 'Heart: Cholesterol',
                'control': control,
                'treatment': treatment,
                'description': 'Healthy vs Heart Disease (Cholesterol)'
            })

        # Max heart rate comparison
        control = df[df['diagnosis'] == 0]['thalach'].dropna().values
        treatment = df[df['diagnosis'] == 1]['thalach'].dropna().values

        control = control[(control > 60) & (control < 220)]
        treatment = treatment[(treatment > 60) & (treatment < 220)]

        if len(control) > 20 and len(treatment) > 20:
            studies.append({
                'name': 'Heart: Max Heart Rate',
                'control': control,
                'treatment': treatment,
                'description': 'Healthy vs Heart Disease (Max Heart Rate)'
            })

        # Resting blood pressure
        control = df[df['diagnosis'] == 0]['trestbps'].dropna().values
        treatment = df[df['diagnosis'] == 1]['trestbps'].dropna().values

        control = control[(control > 80) & (control < 200)]
        treatment = treatment[(treatment > 80) & (treatment < 200)]

        if len(control) > 20 and len(treatment) > 20:
            studies.append({
                'name': 'Heart: Resting BP',
                'control': control,
                'treatment': treatment,
                'description': 'Healthy vs Heart Disease (Resting BP)'
            })

    # 3. Wine Dataset - Compare red vs white
    if 'wine_red' in datasets and 'wine_white' in datasets:
        # Alcohol content
        control = datasets['wine_red']['alcohol'].values
        treatment = datasets['wine_white']['alcohol'].values

        control = control[(control > 8) & (control < 14)]
        treatment = treatment[(treatment > 8) & (treatment < 14)]

        if len(control) > 20 and len(treatment) > 20:
            studies.append({
                'name': 'Wine: Alcohol',
                'control': control,
                'treatment': treatment,
                'description': 'Red vs White Wine (Alcohol content)'
            })

        # pH level
        control = datasets['wine_red']['pH'].values
        treatment = datasets['wine_white']['pH'].values

        control = control[(control > 2.5) & (control < 4.5)]
        treatment = treatment[(treatment > 2.5) & (treatment < 4.5)]

        if len(control) > 20 and len(treatment) > 20:
            studies.append({
                'name': 'Wine: pH',
                'control': control,
                'treatment': treatment,
                'description': 'Red vs White Wine (pH level)'
            })

    return studies


def run_ipd_qma_analysis(studies: list):
    """Run IPD-QMA analysis on the studies."""
    print("\n" + "=" * 70)
    print("IPD-QMA ANALYSIS WITH REAL UCI DATASETS")
    print("=" * 70)

    # Convert to format needed by IPDQMA
    studies_data = [(s['control'], s['treatment']) for s in studies]

    print(f"\nAnalyzing {len(studies)} treatment/control comparisons:")
    for i, s in enumerate(studies):
        print(f"  {i+1}. {s['name']}: n={len(s['control'])} vs {len(s['treatment'])}")

    # Configure IPD-QMA
    config = IQMAConfig(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        n_bootstrap=500,
        use_random_effects=True,
        tau2_estimator='dl',
        confidence_level=0.95
    )

    # Run analysis
    analyzer = IPDQMA(config)
    results = analyzer.fit(studies_data)

    # Display results
    summary = analyzer.summary()

    # Generate plots
    print("\nGenerating plots...")

    # Fan plot
    fig1 = analyzer.plot(figsize=(12, 6))
    plt.savefig('ipd_qma_real_data_fan_plot.png', dpi=150, bbox_inches='tight')
    print("  Saved: ipd_qma_real_data_fan_plot.png")

    # Forest plot
    fig2 = analyzer.plot_forest(figsize=(10, 8))
    plt.savefig('ipd_qma_real_data_forest_plot.png', dpi=150, bbox_inches='tight')
    print("  Saved: ipd_qma_real_data_forest_plot.png")

    # Export results
    try:
        analyzer.export_results('ipd_qma_real_data_results.xlsx', format='xlsx')
        print("  Saved: ipd_qma_real_data_results.xlsx")
    except Exception as e:
        print(f"  Note: Excel export skipped ({e})")

    return analyzer, results


def create_individual_study_analyses(studies: list):
    """Create individual analyses for each comparison."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL STUDY ANALYSES")
    print("=" * 70)

    config = IQMAConfig(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        n_bootstrap=200,
        use_random_effects=False  # Fixed effect for single studies
    )

    for i, study in enumerate(studies):
        print(f"\n--- {study['name']} ---")
        print(f"Description: {study['description']}")
        print(f"Control: n={len(study['control'])}, mean={np.mean(study['control']):.2f}, sd={np.std(study['control']):.2f}")
        print(f"Treatment: n={len(study['treatment'])}, mean={np.mean(study['treatment']):.2f}, sd={np.std(study['treatment']):.2f}")

        # Single study analysis
        analyzer = IPDQMA(config)
        result = analyzer.analyze_study(study['control'], study['treatment'])

        print(f"Effect at quantiles:")
        for j, q in enumerate(config.quantiles):
            print(f"  Q{q*100:.0f}: {result['quantiles'][j]:.2f} (SE={result['se_quantiles'][j]:.2f})")

        print(f"Slope (Q90-Q10): {result['slope']:.2f} (SE={result['se_slope']:.2f})")
        print(f"lnVR (variance ratio): {result['lnvr']:.4f} (SE={result['se_lnvr']:.4f})")


def main():
    """Main execution."""
    print("IPD-QMA with Real UCI Datasets")
    print("=" * 70)

    # Load data
    print("\n[*] Loading datasets...")
    datasets = load_uci_data()

    if not datasets:
        print("No datasets found. Run fetch_real_ipd.py first.")
        return

    # Create studies
    print("\n[*] Creating treatment/control comparisons...")
    studies = create_studies_from_datasets(datasets)
    print(f"Created {len(studies)} comparisons")

    # Show individual study statistics
    create_individual_study_analyses(studies)

    # Run IPD-QMA meta-analysis
    analyzer, results = run_ipd_qma_analysis(studies)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Show key findings
    print("\nKey Findings:")
    print(f"  Slope Test P-value: {results['slope_test']['p']:.4f}")
    print(f"  Interpretation: {results['slope_test']['interpretation']}")
    print(f"\n  lnVR Test P-value: {results['lnvr_test']['p']:.4f}")
    print(f"  Interpretation: {results['lnvr_test']['interpretation']}")

    plt.show()

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
