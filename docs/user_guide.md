# IPD-QMA User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Preparation](#data-preparation)
5. [Basic Usage](#basic-usage)
6. [Configuration Options](#configuration-options)
7. [Interpreting Results](#interpreting-results)
8. [Visualizations](#visualizations)
9. [Exporting Results](#exporting-results)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

---

## Introduction

**IPD-QMA** (Individual Participant Data Quantile Meta-Analysis) is a statistical method for detecting heterogeneous treatment effects across patient severity distributions. Unlike traditional meta-analysis which focuses on mean differences, IPD-QMA examines treatment effects across multiple quantiles of the outcome distribution.

### What is Heterogeneous Treatment Effect?

Heterogeneous treatment effects occur when a treatment's effectiveness varies across different levels of patient severity or baseline characteristics. For example:
- A treatment may be more effective for severely ill patients than for mildly ill patients
- The treatment effect may increase or decrease across the outcome distribution

### Key Features

- **Quantile-based analysis**: Examines effects at multiple points of the distribution
- **Bootstrap inference**: Provides robust standard errors and confidence intervals
- **Heterogeneity detection**: Tests for location-scale shifts using slope and lnVR statistics
- **Random-effects modeling**: Accounts for between-study variation
- **Parallel processing**: Fast computation for large datasets
- **Progress tracking**: Real-time progress bars for long-running analyses

---

## Installation

### Requirements

- Python 3.8 or higher
- NumPy >= 1.20
- Pandas >= 1.3
- SciPy >= 1.7
- Matplotlib >= 3.3
- tqdm >= 4.60 (optional, for progress bars)

### Install from Source

```bash
cd ipd_qma_project
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For Excel export
pip install openpyxl

# For memory profiling in benchmarks
pip install psutil
```

---

## Quick Start

### Minimal Example

```python
from ipd_qma import IPDQMA, IQMAConfig
import numpy as np

# Prepare your data
control_group = np.array([1.2, 1.5, 1.8, 2.1, 2.4, ...])
treatment_group = np.array([1.8, 2.2, 2.5, 3.0, 3.5, ...])

# Create analyzer
analyzer = IPDQMA()

# Analyze a single study
result = analyzer.analyze_study(control_group, treatment_group)

# For multiple studies (meta-analysis)
studies = [
    (control1, treatment1),
    (control2, treatment2),
    (control3, treatment3)
]
results = analyzer.fit(studies)

# View summary
analyzer.summary()

# Generate plots
analyzer.plot()
analyzer.plot_forest()
```

### Running the Tutorial

```bash
python ipd_qma.py
```

---

## Data Preparation

### Required Data Format

IPD-QMA requires **individual participant data (IPD)** - the raw outcomes for each participant in both control and treatment groups.

#### Data Structure

Each study should provide:
- **Control group outcomes**: Array of continuous outcome values
- **Treatment group outcomes**: Array of continuous outcome values

#### Example Data

```python
# Study 1: Blood pressure reduction
control_1 = [180, 175, 170, 168, 165, ...]  # mmHg
treatment_1 = [160, 155, 150, 148, 145, ...]  # mmHg

# Study 2: Cholesterol levels
control_2 = [240, 235, 230, 225, 220, ...]  # mg/dL
treatment_2 = [210, 200, 195, 190, 185, ...]  # mg/dL
```

### Data Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Sample size per group | 10 | 50+ |
| Number of studies (for meta-analysis) | 1 | 5+ |
| Data type | Continuous | Continuous |
| Missing values | 0% | < 5% |

### Data Quality Checks

```python
# Check for missing values
import numpy as np
if np.any(np.isnan(control_group)) or np.any(np.isnan(treatment_group)):
    raise ValueError("Data contains missing values")

# Check sample size
if len(control_group) < 10:
    print("Warning: Small sample size may produce unstable results")
```

---

## Basic Usage

### Single Study Analysis

```python
from ipd_qma import IPDQMA, IQMAConfig
import numpy as np

# Configure the analysis
config = IQMAConfig(
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    n_bootstrap=500,
    confidence_level=0.95
)

# Create analyzer
analyzer = IPDQMA(config)

# Analyze
result = analyzer.analyze_study(control, treatment)

# Access results
print(f"Effect at median (Q50): {result['quantiles'][2]:.3f}")
print(f"Slope (Q90-Q10): {result['slope']:.3f}")
print(f"lnVR (variance ratio): {result['lnvr']:.3f}")
```

### Multi-Study Meta-Analysis

```python
# Prepare multiple studies
studies = [
    (control1, treatment1),
    (control2, treatment2),
    (control3, treatment3),
    ...
]

# Run meta-analysis
analyzer = IPDQMA()
results = analyzer.fit(studies)

# View results
print(f"Number of studies: {results['n_studies']}")
print(f"Model type: {results['model_type']}")
print(f"\nSlope test p-value: {results['slope_test']['p']:.4f}")
print(f"lnVR test p-value: {results['lnvr_test']['p']:.4f}")
```

---

## Configuration Options

### IQMAConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantiles` | List[float] | [0.1, 0.25, 0.5, 0.75, 0.9] | Quantiles to analyze |
| `n_bootstrap` | int | 200 | Number of bootstrap samples |
| `confidence_level` | float | 0.95 | Confidence level (0-1) |
| `random_seed` | int | None | Random seed for reproducibility |
| `use_random_effects` | bool | True | Use random-effects model |
| `tau2_estimator` | str | 'dl' | Heterogeneity estimator ('dl', 'pm') |
| `n_workers` | int | None | Number of parallel workers |
| `show_progress` | bool | True | Show progress bars |
| `parallel_threshold` | int | 1000 | Min bootstrap for parallel processing |

### Configuration Examples

#### High-Precision Analysis

```python
config = IQMAConfig(
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    n_bootstrap=2000,
    confidence_level=0.99,
    random_seed=42
)
```

#### Fast Analysis (Large Datasets)

```python
config = IQMAConfig(
    quantiles=[0.25, 0.5, 0.75],
    n_bootstrap=100,
    show_progress=False,
    n_workers=4
)
```

#### Fixed-Effect Meta-Analysis

```python
config = IQMAConfig(
    use_random_effects=False,
    tau2_estimator='dl'
)
```

---

## Interpreting Results

### Key Outputs

#### 1. Quantile Effects

Treatment effects at each quantile:
- **Positive values**: Treatment increases outcomes at that quantile
- **Negative values**: Treatment decreases outcomes at that quantile
- **Pattern across quantiles**: Indicates heterogeneity

#### 2. Slope Test

Tests for heterogeneous effects across quantiles:
- **Slope = Q90 effect - Q10 effect**
- **Significant positive slope**: Effects increase with severity
- **Significant negative slope**: Effects decrease with severity
- **Non-significant slope**: Constant effect across distribution

#### 3. lnVR Test (Log Variance Ratio)

Tests for scale shifts:
- **lnVR > 0**: Treatment increases variance
- **lnVR < 0**: Treatment decreases variance
- **lnVR ≈ 0**: No scale change

#### 4. Heterogeneity Statistics

- **I²**: Percentage of variation due to heterogeneity
  - 0-25%: Low heterogeneity
  - 25-50%: Moderate heterogeneity
  - 50-75%: High heterogeneity
  - 75-100%: Very high heterogeneity

- **τ²**: Between-study variance

- **Q-statistic**: Test of heterogeneity (p < 0.05 indicates significant heterogeneity)

### Interpretation Guide

| Scenario | Slope | lnVR | Interpretation |
|----------|-------|------|----------------|
| No effect | Not significant | Not significant | Treatment has no effect |
| Pure location shift | Not significant | Significant | Treatment changes mean but not variance |
| Pure scale shift | Significant | Significant | Treatment changes variance across distribution |
| Location + scale | Significant | Significant | Treatment has complex heterogeneous effects |

---

## Visualizations

### Fan Plot

Shows treatment effects across quantiles:

```python
fig = analyzer.plot(figsize=(12, 6))
plt.savefig('fan_plot.png', dpi=300)
plt.show()
```

**Fan plot elements:**
- Blue line: Pooled effect at each quantile
- Shaded region: 95% confidence interval
- Lighter shaded region: 95% prediction interval (random-effects)
- Gray dots: Individual study effects

### Forest Plot

Shows individual study effects at a specific quantile:

```python
# Plot for median (default)
fig = analyzer.plot_forest()

# Plot for 10th percentile
fig = analyzer.plot_forest(quantile_index=0)

# Customize
fig = analyzer.plot_forest(quantile_index=2, figsize=(10, 8))
plt.savefig('forest_plot.png', dpi=300)
```

---

## Exporting Results

### Excel Export

```python
analyzer.export_results('results.xlsx', format='xlsx')
```

Creates an Excel file with sheets:
- **Summary**: Overall results table
- **Quantile_Profile**: Detailed quantile-by-quantile results
- **Study_Details**: Individual study statistics

### CSV Export

```python
analyzer.export_results('results.csv', format='csv')
```

### Programmatic Access

```python
# Get summary DataFrame
summary = analyzer.summary()

# Access profile data
profile = results['profile']

# Get specific quantile results
q25_effect = profile[profile['Quantile'] == 0.25]['Effect'].values[0]
```

---

## Examples

### Example 1: Clinical Trial Meta-Analysis

```python
from ipd_qma import IPDQMA, IQMAConfig
import pandas as pd

# Load trial data
trials = pd.read_csv('clinical_trials.csv')

# Convert to IPD-QMA format
studies = []
for trial_id in trials['trial_id'].unique():
    trial_data = trials[trials['trial_id'] == trial_id]
    control = trial_data[trial_data['treatment'] == 0]['outcome'].values
    treatment = trial_data[trial_data['treatment'] == 1]['outcome'].values
    studies.append((control, treatment))

# Configure and analyze
config = IQMAConfig(
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    n_bootstrap=1000,
    use_random_effects=True,
    tau2_estimator='dl'
)

analyzer = IPDQMA(config)
results = analyzer.fit(studies)

# Interpret results
print("Slope Test:", results['slope_test']['interpretation'])
print("lnVR Test:", results['lnvr_test']['interpretation'])

# Generate plots
analyzer.plot()
analyzer.plot_forest()

# Export
analyzer.export_results('trial_analysis.xlsx')
```

### Example 2: Reproducible Analysis

```python
from ipd_qma import IPDQMA, IQMAConfig
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

config = IQMAConfig(
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    n_bootstrap=500,
    random_seed=42,
    use_random_effects=True
)

analyzer = IPDQMA(config)
results = analyzer.fit(studies)

# Results will be identical when run again
```

### Example 3: High-Performance Analysis

```python
from ipd_qma import IPDQMA, IQMAConfig

# Configure for maximum performance
config = IQMAConfig(
    n_bootstrap=2000,
    n_workers=None,  # Use all available CPUs
    parallel_threshold=500,  # Use parallel processing for 500+ bootstraps
    show_progress=True
)

analyzer = IPDQMA(config)
results = analyzer.fit(large_dataset)
```

---

## Troubleshooting

### Common Issues

#### Issue: "Small sample size" warning

**Solution:**
- Increase sample size if possible
- Use fewer quantiles
- Increase bootstrap samples (but results may still be unstable)

#### Issue: Slow analysis

**Solutions:**
- Reduce `n_bootstrap`
- Use fewer quantiles
- Enable parallel processing: `config = IQMAConfig(n_workers=4)`
- Disable progress bars: `config = IQMAConfig(show_progress=False)`

#### Issue: "Data contains NaN values"

**Solution:**
```python
# Remove missing values
control = control[~np.isnan(control)]
treatment = treatment[~np.isnan(treatment)]
```

#### Issue: Import error for multiprocessing

**Solution:**
- Multiprocessing requires running in `if __name__ == "__main__":` block
- Use `n_workers=1` to disable parallel processing

### Performance Tips

1. **For small datasets (< 100 samples per group)**: Use vectorized bootstrap (default)
2. **For large datasets (> 1000 bootstrap)**: Enable parallel processing
3. **For many studies**: Disable per-study progress bars

---

## FAQ

### Q: What's the minimum sample size?

**A:** Technically 10 per group, but we recommend 50+ for stable results. Small samples may produce unreliable bootstrap estimates.

### Q: Can I use categorical outcomes?

**A:** Currently IPD-QMA only supports continuous outcomes. For categorical data, consider transforming to continuous or using other methods.

### Q: How many quantiles should I use?

**A:**
- 3 quantiles (Q25, Q50, Q75): Quick overview
- 5 quantiles (Q10, Q25, Q50, Q75, Q90): Standard analysis
- 9+ quantiles: Detailed analysis (slower)

### Q: How do I choose between fixed and random effects?

**A:**
- Use **random effects** if you expect between-study variation (default)
- Use **fixed effects** if all studies measure the same population

Check I² statistic:
- I² < 25%: Consider fixed effects
- I² > 50%: Use random effects

### Q: What does a significant slope mean?

**A:** A significant slope indicates that treatment effects vary across the outcome distribution. This suggests:
- The treatment may be more/less effective for different severity levels
- There may be a scale shift (variance change)

### Q: Can I combine IPD-QMA with traditional meta-analysis?

**A:** Yes! IPD-QMA complements traditional meta-analysis by:
- Detecting heterogeneous effects missed by mean-difference methods
- Providing insights into effect patterns across distributions

---

## Citation

If you use IPD-QMA in your research, please cite:

```
IPD-QMA: Individual Participant Data Quantile Meta-Analysis
Version 2.0
https://github.com/yourusername/ipd-qma
```

---

## Support

For issues, questions, or contributions:
- GitHub: https://github.com/yourusername/ipd-qma/issues
- Email: your-email@example.com

---

*Last updated: 2024*
