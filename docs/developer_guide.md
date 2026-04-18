# IPD-QMA Developer Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Development Setup](#development-setup)
4. [Code Architecture](#code-architecture)
5. [Testing](#testing)
6. [Benchmarking](#benchmarking)
7. [Adding Features](#adding-features)
8. [Documentation](#documentation)
9. [Release Process](#release-process)
10. [Contributing](#contributing)

---

## Introduction

This guide is for developers who want to:
- Contribute to IPD-QMA
- Understand the codebase architecture
- Add new features
- Fix bugs
- Run tests and benchmarks

### Development Principles

1. **Simplicity**: Keep code simple and readable
2. **Performance**: Optimize for speed, especially bootstrap computation
3. **Reproducibility**: Ensure results are reproducible with same seed
4. **Documentation**: Document all public APIs
5. **Testing**: Maintain high test coverage (> 80%)

---

## Project Structure

```
ipd_qma_project/
├── ipd_qma.py                  # Main package (core algorithms)
├── ipd_qma_advanced.py         # Advanced statistical methods (planned)
├── ipd_qma_plots.py            # Enhanced visualizations (planned)
├── ipd_qma_validation.py       # Data validation (planned)
├── run_ipd_qma_real_data.py    # Real data analysis script
├── fetch_real_ipd.py           # Data fetching utilities
├── data_loader.py              # Data loading utilities
│
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_ipd_qma.py         # Unit and integration tests
│
├── benchmarks/                 # Performance benchmarks
│   ├── __init__.py
│   └── benchmark_ipd_qma.py    # Benchmark suite
│
├── docs/                       # Documentation
│   ├── user_guide.md           # User documentation
│   ├── developer_guide.md      # This file
│   └── api_reference.md        # API documentation
│
├── data/                       # Example datasets
│   ├── diabetes.csv
│   ├── heart_disease.csv
│   └── ...
│
├── docs/                       # Additional documentation
│   ├── DATA_REQUIREMENTS.md
│   └── README.md
│
├── setup.py                    # Package setup (for PyPI)
├── requirements.txt            # Python dependencies
├── .github/workflows/          # CI/CD workflows
└── README.md                   # Project overview
```

---

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (optional but recommended)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ipd-qma.git
cd ipd-qma
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

4. **Install in development mode**
```bash
pip install -e .
```

5. **Run tests**
```bash
pytest tests/
```

### Development Dependencies

Create `requirements-dev.txt`:

```
# Testing
pytest>=7.0
pytest-cov>=4.0

# Code quality
black>=22.0
flake8>=5.0
mypy>=1.0

# Documentation
sphinx>=5.0
sphinx-rtd-theme>=1.0

# Performance
psutil>=5.9
```

---

## Code Architecture

### Core Classes

#### 1. IQMAConfig (Configuration)

```python
@dataclass
class IQMAConfig:
    quantiles: List[float]
    n_bootstrap: int
    confidence_level: float
    random_seed: Optional[int]
    use_random_effects: bool
    tau2_estimator: str
    n_workers: Optional[int]
    show_progress: bool
    parallel_threshold: int
```

**Key design decisions:**
- Uses `@dataclass` for clean configuration
- All parameters have sensible defaults
- Immutable after initialization (prevents accidental modification)

#### 2. IPDQMA (Main Analyzer)

```python
class IPDQMA:
    def __init__(self, config: Optional[IQMAConfig] = None, ...)
    def analyze_study(self, control, treatment) -> Dict
    def fit(self, studies_data) -> Dict
    def plot(self) -> Figure
    def plot_forest(self) -> Figure
    def summary(self) -> DataFrame
    def export_results(self, filepath) -> None
```

**Key design decisions:**
- Separates single-study analysis from meta-analysis
- Results stored in `self.results` after `fit()`
- Visualization methods create new figures (don't modify state)
- Export methods work on fitted results

### Algorithm Flow

```
1. User creates IPDQMA instance with config
2. User calls fit() with study data
3. For each study:
   a. analyze_study() performs bootstrap quantile estimation
   b. Returns quantile effects, slope, lnVR, SEs
4. For each quantile:
   a. Pool study effects using FE or RE model
   b. Estimate heterogeneity (τ², I²)
   c. Calculate CIs and prediction intervals
5. Pool slope and lnVR across studies
6. Store results in self.results
7. User can now call plot(), summary(), export_results()
```

### Bootstrap Implementation

**Vectorized (default, < 1000 bootstraps)**
```python
idx = np.random.randint(0, n, (n_boot, n))
boot_data = data[idx]
quantiles = np.percentile(boot_data, [q*100 for q in quantiles], axis=1)
```

**Parallel (>= 1000 bootstraps)**
```python
with Pool(n_workers) as pool:
    chunks = split_work(n_boot, n_workers)
    results = pool.map(_bootstrap_worker, chunks)
```

---

## Testing

### Test Structure

Tests are organized by class/function:

```python
class TestIQMAConfig:
    """Test configuration class"""
    def test_default_configuration()
    def test_custom_configuration()
    ...

class TestIPDQMABasics:
    """Test basic initialization"""
    def test_initialization_with_config()
    ...

class TestAnalyzeStudy:
    """Test single study analysis"""
    def test_analyze_study_returns_correct_keys()
    def test_quantile_estimation()
    ...
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ipd_qma.py

# Run specific test class
pytest tests/test_ipd_qma.py::TestAnalyzeStudy

# Run with coverage
pytest --cov=ipd_qma tests/

# Verbose output
pytest -v tests/

# Stop on first failure
pytest -x tests/
```

### Writing Tests

**Test naming convention:**
- Test classes: `Test<ClassName>`
- Test functions: `test_<method>_<condition>`

**Example test:**

```python
def test_quantile_estimation_with_normal_data(self, normal_data):
    """Test that quantiles are estimated correctly."""
    control, treatment = normal_data
    analyzer = IPDQMA(quantiles=[0.25, 0.5, 0.75])
    result = analyzer.analyze_study(control, treatment)

    assert len(result['quantiles']) == 3
    assert np.all(np.abs(result['quantiles']) < 10)  # Reasonable range
```

### Test Fixtures

```python
@pytest.fixture
def normal_data():
    """Generate normally distributed test data."""
    np.random.seed(42)
    control = np.random.normal(0, 1, 100)
    treatment = np.random.normal(0.5, 1, 100)
    return control, treatment

@pytest.fixture
def multi_study_data():
    """Generate data for multiple studies."""
    np.random.seed(42)
    studies = []
    for i in range(10):
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.5, 1, 100)
        studies.append((control, treatment))
    return studies
```

### Coverage Goals

- Target: > 80% code coverage
- Critical paths: 100% coverage
- Edge cases: Explicit tests

---

## Benchmarking

### Running Benchmarks

```bash
# Quick benchmark
python benchmarks/benchmark_ipd_qma.py --quick

# Full benchmark suite
python benchmarks/benchmark_ipd_qma.py

# Custom output directory
python benchmarks/benchmark_ipd_qma.py --output my_results/
```

### Benchmark Categories

1. **Scalability**: Number of studies (5, 10, 20, 50)
2. **Bootstrap**: Bootstrap samples (100, 500, 1000, 2000)
3. **Sample size**: Samples per group (50, 100, 200, 500)
4. **Parallel vs Sequential**: Performance comparison
5. **Quantiles**: Different quantile configurations (3, 5, 9, 19)

### Performance Targets

| Scenario | Target | Notes |
|----------|--------|-------|
| 10 studies, 1000 bootstrap | < 30s | Standard analysis |
| 50 studies, 500 bootstrap | < 2 min | Large meta-analysis |
| 100 studies | < 500MB memory | Memory usage |

### Adding Benchmarks

```python
def benchmark_new_feature(self):
    """Benchmark a new feature."""
    studies = self._generate_test_data(10, 100)

    def run():
        # Your feature here
        pass

    result = self._run_benchmark(
        "New Feature",
        run,
        {'n_studies': 10, 'n_bootstrap': 500}
    )
    self.results.append(result)
```

---

## Adding Features

### Adding a New Quantile Estimator

1. **Add to `analyze_study` method:**

```python
def analyze_study(self, control, treatment):
    # ... existing code ...

    # Add your new estimator
    new_estimate = self._my_new_estimator(control, treatment)

    return {
        # ... existing returns ...
        'new_estimate': new_estimate
    }
```

2. **Add private method:**

```python
def _my_new_estimator(self, control, treatment):
    """Implement new estimation method."""
    # Your implementation
    return estimate
```

3. **Add tests:**

```python
class TestNewEstimator:
    def test_new_estimator_returns_valid_result(self):
        analyzer = IPDQMA()
        result = analyzer.analyze_study(control, treatment)
        assert 'new_estimate' in result
        assert isinstance(result['new_estimate'], (int, float))
```

### Adding a New Visualization

1. **Create method in IPDQMA:**

```python
def plot_my_visualization(self, **kwargs) -> plt.Figure:
    """Create a new visualization."""
    if self.results is None:
        raise ValueError("Run fit() first.")

    fig, ax = plt.subplots(**kwargs)

    # Your plotting code here

    return fig
```

2. **Add test:**

```python
def test_plot_my_visualization_returns_figure(self, fitted_analyzer):
    fig = fitted_analyzer.plot_my_visualization()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

### Adding a New Heterogeneity Estimator

1. **Update `_estimate_heterogeneity`:**

```python
def _estimate_heterogeneity(self, estimates, se, method='dl'):
    # ... existing code ...

    elif method == 'my_method':
        tau2 = self._my_heterogeneity_method(estimates, se)

    # ... rest of method ...
```

2. **Implement method:**

```python
def _my_heterogeneity_method(self, estimates, se):
    """Implement new heterogeneity estimator."""
    # Your implementation
    return tau2
```

3. **Update config validation:**

```python
def __post_init__(self):
    if self.quantiles is None:
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    # Add validation
    if self.tau2_estimator not in ['dl', 'pm', 'my_method']:
        raise ValueError(f"Unknown estimator: {self.tau2_estimator}")
```

---

## Documentation

### Documentation Structure

```
docs/
├── user_guide.md         # For end users
├── developer_guide.md    # This file
├── api_reference.md      # API documentation
└── examples/             # Jupyter notebook tutorials
    ├── basic_usage.ipynb
    ├── advanced_analysis.ipynb
    └── real_data_example.ipynb
```

### Docstring Style

Use NumPy-style docstrings:

```python
def analyze_study(
    self,
    control: Union[np.ndarray, List],
    treatment: Union[np.ndarray, List]
) -> Dict[str, np.ndarray]:
    """
    Analyze a single study using bootstrap quantile estimation.

    Parameters
    ----------
    control : array-like
        Control group outcomes
    treatment : array-like
        Treatment group outcomes

    Returns
    -------
    dict
        Dictionary containing effect estimates and standard errors

    Examples
    --------
    >>> analyzer = IPDQMA()
    >>> result = analyzer.analyze_study([1,2,3], [2,3,4])
    >>> print(result['quantiles'])
    """
```

### Generating API Docs

With Sphinx:

```bash
cd docs
sphinx-apidoc -o api ../ipd_qma_project
make html
```

---

## Release Process

### Version Numbering

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Pre-Release Checklist

1. [ ] All tests pass
2. [ ] Code coverage > 80%
3. [ ] Benchmarks meet performance targets
4. [ ] Documentation updated
5. [ ] Changelog updated
6. [ ] Version number updated

### Release Steps

```bash
# 1. Update version in __init__.py
# 2. Update CHANGELOG.md
# 3. Commit changes
git add .
git commit -m "Release v2.0.0"

# 4. Create tag
git tag v2.0.0

# 5. Push to GitHub
git push origin main --tags

# 6. Build PyPI package
python setup.py sdist bdist_wheel

# 7. Upload to PyPI
twine upload dist/*

# 8. Create GitHub release with notes
```

---

## Contributing

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
```bash
git checkout -b feature/my-feature
```

3. **Make your changes**
4. **Add tests**
5. **Run tests and benchmarks**
6. **Update documentation**
7. **Submit a pull request**

### Pull Request Guidelines

- Describe your changes clearly
- Reference related issues
- Include tests for new features
- Ensure all tests pass
- Update documentation
- Follow code style (Black formatting)

### Code Review Process

1. Automated checks run (tests, linting)
2. Maintainer reviews code
3. Feedback addressed
4. Approval and merge

### Code Style

```bash
# Format code with Black
black ipd_qma_project/

# Check linting with Flake8
flake8 ipd_qma_project/

# Type checking with mypy
mypy ipd_qma_project/
```

---

## Troubleshooting Development Issues

### ImportError in Multiprocessing

**Problem:** Multiprocessing fails on Windows

**Solution:** Use `if __name__ == "__main__":` guard

```python
if __name__ == "__main__":
    # Your code here
    pass
```

### Test Failures

**Common causes:**
1. Random seed not set (use `np.random.seed(42)` in fixtures)
2. Wrong matplotlib backend (use `matplotlib.use('Agg')` in tests)
3. File path issues (use `os.path` for cross-platform compatibility)

### Performance Regression

**Debug steps:**
1. Run benchmarks on previous version
2. Run benchmarks on new version
3. Compare results
4. Profile with `cProfile` if needed

```python
import cProfile
cProfile.run('analyzer.fit(studies)', 'output.prof')
```

---

## Resources

### External Dependencies

- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data structures
- [SciPy](https://scipy.org/) - Statistical functions
- [Matplotlib](https://matplotlib.org/) - Plotting
- [tqdm](https://tqdm.github.io/) - Progress bars

### Statistical References

- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177-188.
- Paule, R. C., & Mandel, J. (1982). Consensus values and weighting factors. *J. Res. Natl. Bur. Stand.*, 87(5), 377-385.

### Related Packages

- [metafor](https://www.metafor-project.org/) - R package for meta-analysis
- [meta](https://cran.r-project.org/web/packages/meta/index.html) - R package for meta-analysis

---

*Last updated: 2024*
