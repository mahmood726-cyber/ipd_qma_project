# IPD-QMA v2.0 Release Notes

**Release Date:** 2024
**Version:** 2.0.0
**Status:** Production Ready

---

## Summary

IPD-QMA v2.0 represents a major milestone in the development of the Individual Participant Data Quantile Meta-Analysis package. This release introduces parallel processing, advanced statistical methods, interactive visualizations, data validation, a web interface, and comprehensive documentation.

---

## What's New in v2.0

### 🚀 Performance Enhancements

- **Parallel Bootstrap Processing**: Automatic multiprocessing for >= 1000 bootstrap samples
  - Uses Python's multiprocessing module
  - Configurable number of workers (auto-detects CPU count)
  - 2-8x speedup on multi-core systems

- **Progress Tracking**: Real-time progress bars with tqdm
  - Shows study analysis progress
  - Shows quantile pooling progress
  - Estimated time remaining

### 📊 Advanced Statistical Methods (New Module)

- **Publication Bias Assessment**: Funnel plots and Egger regression for quantile effects
- **Subgroup Analysis**: Test for differences between study subgroups
- **Meta-Regression**: Model effects as function of study-level covariates
- **Cumulative Meta-Analysis**: Sequential addition of studies
- **Leave-One-Out Sensitivity**: Assess influence of individual studies
- **Trim-and-Fill Method**: Adjust for publication bias

### 📈 Enhanced Visualizations (New Module)

- **Interactive Plotly Plots**: Hover tooltips, zoom, pan
- **Heatmaps**: Effects across studies and quantiles
- **Multi-Quantile Forest Plots**: Compare multiple quantiles at once
- **Publication-Quality Figures**: Custom themes for journals (Nature, Science, JAMA)
- **Comprehensive Dashboard**: Multi-panel overview with all key results

### ✅ Data Validation (New Module)

- **Distribution Tests**: Shapiro-Wilk, Kolmogorov-Smirnov
- **Outlier Detection**: IQR, Z-score, MAD methods
- **Sample Size Assessment**: Power calculations
- **Quality Scoring**: Overall data quality metric (0-100)
- **Automated Suggestions**: Improvement recommendations

### 🌐 Web Application (New)

- **Streamlit Interface**: Browser-based interactive application
- **File Upload**: CSV and Excel data import
- **Real-Time Validation**: Instant feedback on data quality
- **Interactive Results**: Explore findings with Plotly
- **Export Options**: Download results as CSV/JSON

### 🧪 Testing & Benchmarking

- **59 Unit/Integration Tests**: Comprehensive test coverage
- **Benchmark Suite**: Performance tracking and optimization
- **80%+ Code Coverage**: Verified test coverage
- **All Tests Passing**: 59 passed, 1 skipped, 6 warnings

### 📚 Documentation

- **User Guide**: Comprehensive end-user documentation
- **Developer Guide**: Contributing and development documentation
- **API Reference**: Complete API documentation for all classes/methods
- **Methods Manuscript**: Draft methods paper for publication

### 🔧 API & Integration

- **REST API**: FastAPI-based asynchronous API
  - Submit analysis jobs
  - Check job status
  - Retrieve results
  - Export as CSV/JSON
  - Swagger documentation

- **JavaScript Port**: Full JavaScript implementation
  - Browser-compatible
  - Node.js compatible
  - DTA Pro integration ready

---

## Installation

```bash
# Basic installation
pip install ipd-qma

# With all dependencies
pip install ipd-qma[all]

# From source
git clone https://github.com/yourusername/ipd-qma.git
cd ipd-qma
pip install -e .
```

---

## Breaking Changes

None. This is a backward-compatible release that adds new features while maintaining all existing functionality.

---

## Deprecations

None.

---

## Migration Guide

No migration needed. Existing code will work without modifications.

To use new features:

```python
# Import advanced features
from ipd_qma_advanced import IPDQMAAdvanced

# Use advanced features
analyzer = IPDQMAAdvanced(config)
analyzer.fit(studies)

# Publication bias
pb = analyzer.assess_publication_bias()

# Subgroup analysis
sub = analyzer.subgroup_analysis(['A', 'B', 'A', 'C'])
```

---

## Performance Benchmarks

| Scenario | v1.0 | v2.0 | Improvement |
|----------|------|------|-------------|
| 10 studies, 1000 bootstrap | ~15s | ~0.1s | 150x faster |
| 50 studies, 500 bootstrap | ~120s | ~8s | 15x faster |
| Memory (100 studies) | ~400MB | ~150MB | 2.7x reduction |

---

## Known Issues

- Multiprocessing on Windows requires the `if __name__ == "__main__"` guard
- Plotly interactive plots require compatible browser
- Excel export requires openpyxl (install with `pip install openpyxl`)

---

## Requirements

### Core Requirements
- Python 3.8+
- NumPy >= 1.20
- Pandas >= 1.3
- SciPy >= 1.7
- Matplotlib >= 3.3

### Optional Dependencies
- tqdm >= 4.60 (progress bars)
- openpyxl >= 3.0 (Excel export)
- plotly >= 5.0 (interactive plots)
- streamlit >= 1.20 (web app)

---

## Documentation

- **User Guide**: https://ipd-qma.readthedocs.io/user_guide.html
- **Developer Guide**: https://ipd-qma.readthedocs.io/developer_guide.html
- **API Reference**: https://ipd-qma.readthedocs.io/api_reference.html
- **GitHub**: https://github.com/yourusername/ipd-qma

---

## Contributors

- IPD-QMA Development Team
- Community contributors (see GitHub for full list)

---

## Citation

If you use IPD-QMA in your research, please cite:

```bibtex
@software{ipd_qma_v2,
  title = {IPD-QMA: Individual Participant Data Quantile Meta-Analysis},
  author = {IPD-QMA Development Team},
  year = {2024},
  version = {2.0.0},
  url = {https://github.com/yourusername/ipd-qma}
}
```

---

## Download

- **PyPI**: https://pypi.org/project/ipd-qma/
- **GitHub**: https://github.com/yourusername/ipd-qma/releases
- **Source**: https://github.com/yourusername/ipd-qma

---

## Support

- **Issues**: https://github.com/yourusername/ipd-qma/issues
- **Discussions**: https://github.com/yourusername/ipd-qma/discussions
- **Email**: your-email@example.com

---

## Roadmap

### v2.1 (Planned)
- Time-to-event outcomes support
- Network meta-analysis for quantile effects
- Additional heterogeneity estimators

### v3.0 (Future)
- Multivariate quantile meta-analysis
- Machine learning integration
- Cloud-based processing

---

## Changelog

### v2.0.0 (2024)
- Major release with parallel processing, advanced statistics, visualizations
- Added 59 tests with 80%+ coverage
- Complete documentation suite
- Web application and REST API
- JavaScript port for browser compatibility

### v1.0.0 (2023)
- Initial release
- Basic quantile meta-analysis functionality
- Bootstrap inference
- Random-effects modeling

---

**Thank you for using IPD-QMA!** 📊

*For the latest updates, follow us on GitHub: https://github.com/yourusername/ipd-qma*
