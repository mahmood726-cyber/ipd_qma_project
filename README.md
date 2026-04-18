# IPD-QMA: Individual Participant Data Quantile Meta-Analysis

**Version 2.0** - A comprehensive Python package for detecting heterogeneous treatment effects across patient severity distributions using quantile-based analysis with bootstrap inference.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-59%20passing-brightgreen.svg)](tests/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

IPD-QMA is a statistical method for detecting **heterogeneous treatment effects** - where treatment effectiveness varies across different levels of patient severity. Unlike traditional meta-analysis that focuses only on mean differences, IPD-QMA examines treatment effects across multiple quantiles of the outcome distribution.

### Key Capabilities

- **Quantile-based analysis**: Examine effects at multiple distribution points (Q10, Q25, Q50, Q75, Q90)
- **Bootstrap inference**: Robust standard errors and confidence intervals
- **Heterogeneity detection**: Slope test and log variance ratio (lnVR) statistics
- **Random-effects modeling**: Account for between-study variation
- **Parallel processing**: Fast computation for large datasets
- **Interactive visualizations**: Both static and interactive Plotly plots
- **Web interface**: Streamlit-based web application
- **Advanced statistics**: Publication bias, subgroup analysis, meta-regression

---

## Features

### Core Analysis (`ipd_qma.py`)

- **Single study analysis**: Bootstrap quantile estimation
- **Multi-study meta-analysis**: Fixed and random-effects models
- **Heterogeneity statistics**: I², Q-statistic, τ² for each quantile
- **Prediction intervals**: Shows where future study effects might fall
- **Bias correction**: BCa (bias-corrected and accelerated) approximation
- **Parallel processing**: Automatic multiprocessing for large bootstrap samples
- **Progress tracking**: Real-time progress bars with tqdm

### Advanced Methods (`ipd_qma_advanced.py`)

- **Publication bias assessment**: Funnel plots, Egger regression for quantiles
- **Subgroup analysis**: Test for differences between study subgroups
- **Meta-regression**: Model effects as function of study-level covariates
- **Cumulative meta-analysis**: Sequential addition of studies
- **Leave-one-out sensitivity**: Assess influence of individual studies
- **Trim-and-fill method**: Adjust for publication bias

### Enhanced Visualizations (`ipd_qma_plots.py`)

- **Interactive plots**: Plotly-based interactive visualizations
- **Fan plots**: Treatment effects across quantiles with CI/PI
- **Forest plots**: Study effects at specific quantiles
- **Heatmaps**: Effects across studies and quantiles
- **Publication-quality figures**: Custom themes for journals
- **Comprehensive dashboard**: Multi-panel overview

### Data Validation (`ipd_qma_validation.py`)

- **Distribution tests**: Normality tests (Shapiro-Wilk, KS test)
- **Outlier detection**: IQR, Z-score, and MAD methods
- **Sample size assessment**: Power calculations and adequacy checks
- **Quality scoring**: Overall data quality metric (0-100)
- **Improvement suggestions**: Automated recommendations

### Web Application (`web_app/app.py`)

- **File upload**: CSV/Excel data import
- **Manual input**: Interactive data entry
- **Example datasets**: Built-in demonstration data
- **Real-time validation**: Instant feedback on data quality
- **Interactive results**: Explore findings with Plotly
- **Export options**: Download results as CSV/JSON

### Testing & Benchmarking

- **Comprehensive test suite**: 59 unit and integration tests
- **Benchmark suite**: Performance tracking and optimization
- **Code coverage**: 80%+ coverage target

---

## Installation

### From Source

```bash
git clone https://github.com/yourusername/ipd-qma.git
cd ipd-qma
pip install -e .
```

### With All Dependencies

```bash
pip install -e ".[all]"
```

### Specific Components

```bash
# Core only
pip install ipd-qma

# With interactive plots
pip install ipd-qma[plots]

# With progress bars
pip install ipd-qma[progress]

# With web app
pip install ipd-qma[web]

# Development
pip install -e ".[dev]"
```

### Requirements

- Python 3.8 or higher
- NumPy >= 1.20
- Pandas >= 1.3
- SciPy >= 1.7
- Matplotlib >= 3.3

---

## Quick Start

### Basic Usage

```python
from ipd_qma import IPDQMA, IQMAConfig
import numpy as np

# Prepare data (control and treatment outcomes for each study)
studies = [
    (control1, treatment1),
    (control2, treatment2),
    (control3, treatment3)
]

# Configure analysis
config = IQMAConfig(
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    n_bootstrap=500,
    use_random_effects=True,
    random_seed=42
)

# Run analysis
analyzer = IPDQMA(config)
results = analyzer.fit(studies)

# View results
analyzer.summary()

# Generate plots
analyzer.plot()
analyzer.plot_forest()

# Export
analyzer.export_results('results.xlsx')
```

### Advanced Analysis

```python
from ipd_qma_advanced import IPDQMAAdvanced

# Use advanced features
analyzer = IPDQMAAdvanced(config)
analyzer.fit(studies)

# Publication bias assessment
pb = analyzer.assess_publication_bias()

# Subgroup analysis
subgroups = ['A', 'B', 'A', 'C']  # Group labels for each study
sub = analyzer.subgroup_analysis(subgroups)

# Leave-one-out sensitivity
loo = analyzer.leave_one_out()
```

### Web Application

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

---

## Documentation

- **[User Guide](docs/user_guide.md)**: Comprehensive documentation for end users
- **[Developer Guide](docs/developer_guide.md)**: Contributing and development
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Data Requirements](DATA_REQUIREMENTS.md)**: Data format specifications

---

## Project Structure

```
ipd_qma_project/
├── ipd_qma.py                 # Core analysis module
├── ipd_qma_advanced.py        # Advanced statistical methods
├── ipd_qma_plots.py           # Enhanced visualizations
├── ipd_qma_validation.py      # Data validation
│
├── web_app/
│   └── app.py                 # Streamlit web application
│
├── tests/
│   ├── __init__.py
│   └── test_ipd_qma.py        # Test suite (59 tests)
│
├── benchmarks/
│   ├── __init__.py
│   └── benchmark_ipd_qma.py   # Performance benchmarks
│
├── docs/
│   ├── user_guide.md
│   ├── developer_guide.md
│   └── api_reference.md
│
├── data/                      # Example datasets
├── setup.py                   # PyPI package configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Performance

IPD-QMA is optimized for performance with parallel processing and vectorized operations.

### Benchmarks

| Scenario | Target | Typical Performance |
|----------|--------|---------------------|
| 10 studies, 1000 bootstrap | < 30s | ~0.1-5s |
| 50 studies, 500 bootstrap | < 2 min | ~5-30s |
| 100 studies | < 500MB RAM | ~100-200MB |

### Optimization Features

- **Vectorized bootstrap**: Fast computation for < 1000 bootstraps
- **Parallel processing**: Automatic for >= 1000 bootstraps
- **Progress bars**: Real-time feedback with tqdm
- **Memory efficient**: Handles 100+ studies

Run benchmarks:

```bash
python benchmarks/benchmark_ipd_qma.py --quick
```

---

## Statistical Background

IPD-QMA addresses a key limitation of traditional meta-analysis: it can detect **effect heterogeneity** across the outcome distribution.

### Key Statistics

1. **Slope Test**: Tests for trend in effects across quantiles
   - H₀: Effects are constant across quantiles
   - Significant slope = heterogeneous effects

2. **lnVR Test**: Log variance ratio test for scale shifts
   - H₀: Variances are equal
   - Significant lnVR = scale difference

3. **Heterogeneity Statistics**: I², τ², Q-statistic

### Interpretation Guide

| Pattern | Slope | lnVR | Interpretation |
|---------|-------|------|----------------|
| No effect | NS | NS | Treatment has no effect |
| Location shift only | NS | ** | Treatment changes mean |
| Scale shift only | ** | ** | Treatment changes variance |
| Heterogeneous effects | ** | ** | Complex effects vary by severity |

**NS** = Not significant (p ≥ 0.05)
**** = Significant (p < 0.05)

---

## Contributing

We welcome contributions! Please see the [Developer Guide](docs/developer_guide.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/ipd-qma.git
cd ipd-qma
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black ipd_qma_project/

# Run benchmarks
python benchmarks/benchmark_ipd_qma.py
```

---

## Citation

If you use IPD-QMA in your research, please cite:

```bibtex
@software{ipd_qma,
  title = {IPD-QMA: Individual Participant Data Quantile Meta-Analysis},
  author = {IPD-QMA Development Team},
  year = {2024},
  version = {2.0},
  url = {https://github.com/yourusername/ipd-qma}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- Statistical methods based on research in meta-analysis and quantile regression
- Bootstrap implementation following Efron's work
- Inspired by R packages: metafor, meta, metafor

---

## Contact

- **Issues**: https://github.com/yourusername/ipd-qma/issues
- **Email**: your-email@example.com
- **Documentation**: https://ipd-qma.readthedocs.io/

---

*Last updated: 2024*
