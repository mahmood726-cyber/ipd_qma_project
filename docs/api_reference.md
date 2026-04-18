# IPD-QMA API Reference

Complete API documentation for the IPD-QMA package.

---

## Table of Contents

1. [Classes](#classes)
    - [IQMAConfig](#iqmaconfig)
    - [IPDQMA](#ipd-qma)
2. [Functions](#functions)
    - [_bootstrap_worker](#_bootstrap_worker)
3. [Data Structures](#data-structures)
4. [Return Values](#return-values)
5. [Exceptions](#exceptions)

---

## Classes

### IQMAConfig

Configuration class for IPD-QMA analysis.

```python
@dataclass
class IQMAConfig:
    quantiles: List[float] = None
    n_bootstrap: int = 200
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    use_random_effects: bool = True
    tau2_estimator: str = 'dl'
    n_workers: Optional[int] = None
    show_progress: bool = True
    parallel_threshold: int = 1000
```

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantiles` | List[float] | [0.1, 0.25, 0.5, 0.75, 0.9] | Quantiles to analyze (0-1) |
| `n_bootstrap` | int | 200 | Number of bootstrap samples |
| `confidence_level` | float | 0.95 | Confidence level for intervals (0-1) |
| `random_seed` | int | None | Random seed for reproducibility |
| `use_random_effects` | bool | True | Use random-effects model |
| `tau2_estimator` | str | 'dl' | Heterogeneity estimator ('dl', 'pm') |
| `n_workers` | int | None | Number of parallel workers (None=auto) |
| `show_progress` | bool | True | Show progress bars |
| `parallel_threshold` | int | 1000 | Min bootstrap samples for parallel processing |

#### Methods

##### `__post_init__`

```python
def __post_init__(self) -> None
```

Initialize default quantiles if not provided.

**Raises:** None

**Example:**

```python
config = IQMAConfig(quantiles=None)
print(config.quantiles)  # [0.1, 0.25, 0.5, 0.75, 0.9]
```

---

### IPDQMA

Main analyzer class for IPD-QMA.

```python
class IPDQMA:
    def __init__(
        self,
        config: Optional[IQMAConfig] = None,
        quantiles: Optional[List[float]] = None,
        n_boot: int = 200
    )
```

#### Constructor

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | IQMAConfig | None | Configuration object |
| `quantiles` | List[float] | None | Quantiles to analyze (overrides config) |
| `n_boot` | int | 200 | Bootstrap samples (overrides config if no config) |

**Example:**

```python
# With config
config = IQMAConfig(n_bootstrap=500, use_random_effects=True)
analyzer = IPDQMA(config=config)

# With parameters
analyzer = IPDQMA(quantiles=[0.25, 0.5, 0.75], n_boot=500)

# Default
analyzer = IPDQMA()
```

#### Methods

---

##### `analyze_study`

Analyze a single study using bootstrap quantile estimation.

```python
def analyze_study(
    self,
    control: Union[np.ndarray, List],
    treatment: Union[np.ndarray, List],
    show_progress: Optional[bool] = None
) -> Dict[str, np.ndarray]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `control` | array-like | Required | Control group outcomes |
| `treatment` | array-like | Required | Treatment group outcomes |
| `show_progress` | bool | None | Override progress bar setting |

**Returns:**

Dictionary with keys:

| Key | Type | Description |
|-----|------|-------------|
| `quantiles` | ndarray | Effect estimates at each quantile |
| `se_quantiles` | ndarray | Standard errors for quantile effects |
| `quantile_effects_bc` | ndarray | Bias-corrected effect estimates |
| `slope` | float | Slope (Q90 - Q10 effect) |
| `se_slope` | float | Standard error of slope |
| `lnvr` | float | Log variance ratio |
| `se_lnvr` | float | Standard error of lnVR |
| `n_control` | int | Control group sample size |
| `n_treatment` | int | Treatment group sample size |
| `mean_control` | float | Control group mean |
| `mean_treatment` | float | Treatment group mean |
| `sd_control` | float | Control group standard deviation |
| `sd_treatment` | float | Treatment group standard deviation |

**Raises:**

- `ValueError`: If data contains NaN or infinite values
- `UserWarning`: If sample sizes are small (< 10)

**Example:**

```python
analyzer = IPDQMA(n_bootstrap=500)

control = [1.2, 1.5, 1.8, 2.1, 2.4]
treatment = [1.8, 2.2, 2.5, 3.0, 3.5]

result = analyzer.analyze_study(control, treatment)

print(f"Effect at median: {result['quantiles'][2]}")
print(f"Slope: {result['slope']}")
print(f"lnVR: {result['lnvr']}")
```

---

##### `fit`

Fit the IPD-QMA model to multiple studies (meta-analysis).

```python
def fit(
    self,
    studies_data: List[Tuple[Union[np.ndarray, List], Union[np.ndarray, List]]]
) -> Dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `studies_data` | list of tuples | Each tuple: (control_outcomes, treatment_outcomes) |

**Returns:**

Dictionary with keys:

| Key | Type | Description |
|-----|------|-------------|
| `n_studies` | int | Number of studies analyzed |
| `model_type` | str | 'random_effects' or 'fixed_effect' |
| `profile` | DataFrame | Quantile-by-quantile results |
| `slope_test` | dict | Slope test results |
| `lnvr_test` | dict | lnVR test results |
| `study_details` | list | Individual study results |
| `config` | IQMAConfig | Configuration used |

**Slope Test Dictionary:**

| Key | Type | Description |
|-----|------|-------------|
| `estimate` | float | Pooled slope estimate |
| `se` | float | Standard error |
| `p` | float | P-value |
| `ci_lower` | float | Lower CI bound |
| `ci_upper` | float | Upper CI bound |
| `i2` | float | I² heterogeneity (%) |
| `tau2` | float | τ² heterogeneity |
| `q_p` | float | Q-test p-value |
| `interpretation` | str | Textual interpretation |

**lnVR Test Dictionary:** Same structure as slope test.

**Example:**

```python
studies = [
    (control1, treatment1),
    (control2, treatment2),
    (control3, treatment3)
]

analyzer = IPDQMA()
results = analyzer.fit(studies)

print(f"Studies: {results['n_studies']}")
print(f"Model: {results['model_type']}")
print(f"Slope p-value: {results['slope_test']['p']}")
```

---

##### `plot`

Generate fan plot showing treatment effects across quantiles.

```python
def plot(
    self,
    figsize: Tuple[int, int] = (12, 6),
    show_predictions: bool = True
) -> matplotlib.figure.Figure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | tuple | (12, 6) | Figure size (width, height) in inches |
| `show_predictions` | bool | True | Show prediction intervals |

**Returns:**

- `matplotlib.figure.Figure`: The generated figure

**Raises:**

- `ValueError`: If `fit()` has not been called

**Example:**

```python
analyzer = IPDQMA()
analyzer.fit(studies)

# Basic plot
fig = analyzer.plot()

# Custom size, no prediction intervals
fig = analyzer.plot(figsize=(10, 5), show_predictions=False)

plt.savefig('fan_plot.png', dpi=300)
```

---

##### `plot_forest`

Create forest plot for a specific quantile.

```python
def plot_forest(
    self,
    quantile_index: int = -1,
    figsize: Tuple[int, int] = (10, 8)
) -> matplotlib.figure.Figure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantile_index` | int | -1 | Index of quantile to plot (-1 = median) |
| `figsize` | tuple | (10, 8) | Figure size (width, height) |

**Returns:**

- `matplotlib.figure.Figure`: The generated figure

**Raises:**

- `ValueError`: If `fit()` has not been called

**Example:**

```python
analyzer = IPDQMA()
analyzer.fit(studies)

# Plot for median (default)
fig = analyzer.plot_forest()

# Plot for 10th percentile (index 0)
fig = analyzer.plot_forest(quantile_index=0)

# Plot for 90th percentile (index 4)
fig = analyzer.plot_forest(quantile_index=4)
```

---

##### `summary`

Generate summary table of results.

```python
def summary(self) -> pandas.DataFrame
```

**Returns:**

- `pandas.DataFrame`: Summary statistics for each quantile

**Columns:**

| Column | Description |
|--------|-------------|
| `Quantile` | Quantile value |
| `Effect` | Pooled effect estimate |
| `SE` | Standard error |
| `95% CI Lower` | Lower confidence bound |
| `95% CI Upper` | Upper confidence bound |
| `P-value` | P-value |
| `I² (%)` | Heterogeneity percentage |
| `τ²` | Between-study variance |

**Raises:**

- `ValueError`: If `fit()` has not been called

**Example:**

```python
analyzer = IPDQMA()
analyzer.fit(studies)

summary = analyzer.summary()
print(summary)

# Export summary
summary.to_csv('summary.csv')
```

---

##### `export_results`

Export results to file.

```python
def export_results(
    self,
    filepath: str,
    format: str = 'xlsx'
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str | Required | Output file path |
| `format` | str | 'xlsx' | Output format ('xlsx' or 'csv') |

**Raises:**

- `ValueError`: If `fit()` has not been called
- `ImportError`: If format='xlsx' and openpyxl not installed

**Excel sheets (format='xlsx'):**

1. **Summary**: Overall results table
2. **Quantile_Profile**: Detailed quantile-by-quantile results
3. **Study_Details**: Individual study statistics

**Example:**

```python
analyzer = IPDQMA()
analyzer.fit(studies)

# Excel export
analyzer.export_results('results.xlsx', format='xlsx')

# CSV export
analyzer.export_results('results.csv', format='csv')
```

---

#### Private Methods

##### `_validate_inputs`

Validate input data for a single study.

```python
def _validate_inputs(
    self,
    control: np.ndarray,
    treatment: np.ndarray
) -> None
```

##### `_parallel_bootstrap`

Perform bootstrap computation using parallel processing.

```python
def _parallel_bootstrap(
    self,
    control: np.ndarray,
    treatment: np.ndarray,
    show_progress: bool
) -> Tuple[np.ndarray, np.ndarray]
```

##### `_vectorized_bootstrap`

Perform bootstrap computation using vectorized operations.

```python
def _vectorized_bootstrap(
    self,
    control: np.ndarray,
    treatment: np.ndarray,
    show_progress: bool
) -> Tuple[np.ndarray, np.ndarray]
```

##### `_calculate_bias_correction`

Calculate bias correction for bootstrap estimates.

```python
def _calculate_bias_correction(
    self,
    boot_diffs: np.ndarray,
    obs_effects: np.ndarray
) -> np.ndarray
```

##### `_pool_fixed_effect`

Fixed-effect meta-analysis using inverse variance weighting.

```python
def _pool_fixed_effect(
    self,
    estimates: np.ndarray,
    se: np.ndarray
) -> Dict[str, float]
```

**Returns:**

| Key | Description |
|-----|-------------|
| `estimate` | Pooled estimate |
| `se` | Standard error |
| `z` | Z-statistic |
| `p` | P-value |
| `lower` | Lower CI bound |
| `upper` | Upper CI bound |
| `weights` | Study weights |

##### `_estimate_heterogeneity`

Estimate between-study heterogeneity (τ²).

```python
def _estimate_heterogeneity(
    self,
    estimates: np.ndarray,
    se: np.ndarray,
    method: str = 'dl'
) -> Dict[str, float]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | 'dl' | Estimator ('dl' or 'pm') |

**Returns:**

| Key | Description |
|-----|-------------|
| `tau2` | τ² heterogeneity |
| `tau` | τ (sqrt of τ²) |
| `i2` | I² heterogeneity (%) |
| `q` | Q-statistic |
| `q_p` | Q-test p-value |
| `k` | Number of studies |

##### `_pool_random_effects`

Random-effects meta-analysis using DL estimator.

```python
def _pool_random_effects(
    self,
    estimates: np.ndarray,
    se: np.ndarray,
    het: Dict[str, float]
) -> Dict[str, float]
```

**Returns:**

All keys from `_pool_fixed_effect`, plus:

| Key | Description |
|-----|-------------|
| `pred_lower` | Lower prediction bound |
| `pred_upper` | Upper prediction bound |

##### `_interpret_slope`

Interpret the slope test result.

```python
def _interpret_slope(
    self,
    estimate: float,
    p_value: float
) -> str
```

**Returns:** Interpretation string

##### `_interpret_lnvr`

Interpret the lnVR test result.

```python
def _interpret_lnvr(
    self,
    estimate: float,
    p_value: float
) -> str
```

**Returns:** Interpretation string

---

## Functions

### `_bootstrap_worker`

Worker function for parallel bootstrap computation.

```python
def _bootstrap_worker(
    args: Tuple[np.ndarray, np.ndarray, List[float], int]
) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `args` | tuple | (control, treatment, quantiles, n_samples) |

**Returns:**

- `tuple`: (boot_quantiles_control, boot_quantiles_treatment)

**Note:** This is a module-level function used by multiprocessing.

---

## Data Structures

### Study Data Format

Each study is a tuple of two arrays:

```python
study = (control_array, treatment_array)

# Example
study = (
    np.array([1.2, 1.5, 1.8, ...]),  # Control outcomes
    np.array([1.8, 2.2, 2.5, ...])   # Treatment outcomes
)
```

### Studies List Format

Meta-analysis uses a list of studies:

```python
studies = [
    (control1, treatment1),
    (control2, treatment2),
    (control3, treatment3),
    ...
]
```

---

## Return Values

### analyze_study Return Structure

```python
{
    'quantiles': array([q1_effect, q2_effect, ...]),  # Effects at each quantile
    'se_quantiles': array([q1_se, q2_se, ...]),       # Standard errors
    'quantile_effects_bc': array([...]),               # Bias-corrected
    'slope': float,                                    # Q90 - Q10
    'se_slope': float,
    'lnvr': float,                                     # Log variance ratio
    'se_lnvr': float,
    'n_control': int,
    'n_treatment': int,
    'mean_control': float,
    'mean_treatment': float,
    'sd_control': float,
    'sd_treatment': float
}
```

### fit Return Structure

```python
{
    'n_studies': int,
    'model_type': str,                    # 'random_effects' or 'fixed_effect'
    'profile': DataFrame,                 # Quantile profile
    'slope_test': {
        'estimate': float,
        'se': float,
        'p': float,
        'ci_lower': float,
        'ci_upper': float,
        'i2': float,
        'tau2': float,
        'q_p': float,
        'interpretation': str
    },
    'lnvr_test': { ... same structure ... },
    'study_details': [ ... ],             # List of analyze_study results
    'config': IQMAConfig
}
```

### profile DataFrame Structure

| Column | Type | Description |
|--------|------|-------------|
| Quantile | float | Quantile value (0-1) |
| Effect | float | Pooled effect |
| SE | float | Standard error |
| Z | float | Z-statistic |
| P | float | P-value |
| CI_Lower | float | Lower confidence bound |
| CI_Upper | float | Upper confidence bound |
| Pred_Lower | float | Lower prediction bound (if RE) |
| Pred_Upper | float | Upper prediction bound (if RE) |
| I2 | float | I² heterogeneity (%) |
| Tau2 | float | τ² heterogeneity |
| Q | float | Q-statistic |
| Q_P | float | Q-test p-value |

---

## Exceptions

### ValueError

Raised when:

- Data contains NaN values
- Data contains infinite values
- `fit()` not called before `plot()`, `plot_forest()`, `summary()`, or `export_results()`

**Example:**

```python
analyzer = IPDQMA()
control = [1, 2, np.nan, 4]

try:
    result = analyzer.analyze_study(control, treatment)
except ValueError as e:
    print(f"Error: {e}")  # "Data contains NaN values."
```

### UserWarning

Issued when:

- Sample size < 10 (bootstrap may be unstable)

**Example:**

```python
import warnings

warnings.filterwarnings('error')  # Treat warnings as errors

try:
    result = analyzer.analyze_study([1,2,3], [2,3,4])
except UserWarning as e:
    print(f"Warning: {e}")
```

### ImportError

Raised when:

- `export_results()` called with `format='xlsx'` but openpyxl not installed

**Example:**

```python
try:
    analyzer.export_results('results.xlsx', format='xlsx')
except ImportError as e:
    print(f"Install openpyxl: {e}")
```

---

## Type Hints

```python
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Common types
ArrayLike = Union[np.ndarray, List]
StudyData = Tuple[ArrayLike, ArrayLike]
StudiesData = List[StudyData]
```

---

## Usage Examples

### Complete Analysis Pipeline

```python
from ipd_qma import IPDQMA, IQMAConfig
import numpy as np

# 1. Prepare data
studies = []
for i in range(10):
    control = np.random.normal(0, 1, 100)
    treatment = np.random.normal(0.5, 1.2, 100)
    studies.append((control, treatment))

# 2. Configure
config = IQMAConfig(
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    n_bootstrap=1000,
    use_random_effects=True,
    random_seed=42
)

# 3. Analyze
analyzer = IPDQMA(config)
results = analyzer.fit(studies)

# 4. View results
summary = analyzer.summary()
print(summary)

# 5. Visualize
fig1 = analyzer.plot(figsize=(12, 6))
fig2 = analyzer.plot_forest()

# 6. Export
analyzer.export_results('analysis_results.xlsx')
```

---

*Last updated: 2024*
