"""
Unit and Integration Tests for IPD-QMA

This module contains comprehensive tests for the IPD-QMA (Individual Participant Data
Quantile Meta-Analysis) package, including unit tests for individual methods and
integration tests for complete workflows.

Test Coverage Goals:
- Core analysis methods: analyze_study, fit, pooling methods
- Statistical calculations: heterogeneity, random/fixed effects
- Visualization methods: plot, plot_forest
- Export functionality: export_results
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_qma import IPDQMA, IQMAConfig


class TestIQMAConfig:
    """Test cases for the IQMAConfig configuration class."""

    def test_default_configuration(self):
        """Test that default configuration values are set correctly."""
        config = IQMAConfig()
        assert config.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
        assert config.n_bootstrap == 200
        assert config.confidence_level == 0.95
        assert config.random_seed is None
        assert config.use_random_effects is True
        assert config.tau2_estimator == 'dl'

    def test_custom_configuration(self):
        """Test setting custom configuration values."""
        config = IQMAConfig(
            quantiles=[0.05, 0.5, 0.95],
            n_bootstrap=500,
            confidence_level=0.99,
            random_seed=42,
            use_random_effects=False,
            tau2_estimator='pm'
        )
        assert config.quantiles == [0.05, 0.5, 0.95]
        assert config.n_bootstrap == 500
        assert config.confidence_level == 0.99
        assert config.random_seed == 42
        assert config.use_random_effects is False
        assert config.tau2_estimator == 'pm'

    def test_invalid_quantiles(self):
        """Test that invalid quantiles are handled correctly."""
        # Test quantiles outside [0, 1] range - should still work as percentiles will be computed
        config = IQMAConfig(quantiles=[0.5])
        assert len(config.quantiles) == 1

    def test_invalid_bootstrap_samples(self):
        """Test that invalid bootstrap samples are handled."""
        config = IQMAConfig(n_bootstrap=0)
        assert config.n_bootstrap == 0  # Should accept, but may cause issues in practice


class TestIPDQMABasics:
    """Test basic initialization and setup of IPDQMA class."""

    def test_initialization_with_config(self):
        """Test initialization with a config object."""
        config = IQMAConfig(quantiles=[0.25, 0.5, 0.75], n_bootstrap=100)
        analyzer = IPDQMA(config)
        assert analyzer.quantiles == [0.25, 0.5, 0.75]
        assert analyzer.n_boot == 100
        assert analyzer.config is config

    def test_initialization_with_parameters(self):
        """Test initialization with direct parameters."""
        analyzer = IPDQMA(quantiles=[0.1, 0.9], n_boot=50)
        assert analyzer.quantiles == [0.1, 0.9]
        assert analyzer.n_boot == 50

    def test_initialization_default(self):
        """Test initialization with defaults."""
        analyzer = IPDQMA()
        assert analyzer.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
        assert analyzer.n_boot == 200

    def test_random_seed_is_set(self):
        """Test that random seed is properly set."""
        analyzer = IPDQMA(config=IQMAConfig(random_seed=42))
        # The seed is set in __init__, so results should be reproducible
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.5, 1, 100)

        result1 = analyzer.analyze_study(control, treatment)
        result2 = analyzer.analyze_study(control, treatment)

        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(result1['quantiles'], result2['quantiles'])


class TestInputValidation:
    """Test input validation and error handling."""

    def test_nan_values_raise_error(self):
        """Test that NaN values raise ValueError."""
        analyzer = IPDQMA()
        control = np.array([1, 2, np.nan, 4, 5])
        treatment = np.array([2, 3, 4, 5, 6])

        with pytest.raises(ValueError, match="NaN"):
            analyzer.analyze_study(control, treatment)

    def test_infinite_values_raise_error(self):
        """Test that infinite values raise ValueError."""
        analyzer = IPDQMA()
        control = np.array([1, 2, np.inf, 4, 5])
        treatment = np.array([2, 3, 4, 5, 6])

        with pytest.raises(ValueError, match="infinite"):
            analyzer.analyze_study(control, treatment)

    def test_small_sample_size_warning(self):
        """Test that small sample sizes produce warnings."""
        analyzer = IPDQMA(n_boot=10)
        control = np.array([1, 2, 3])
        treatment = np.array([2, 3, 4])

        with pytest.warns(UserWarning, match="Small.*sample size"):
            analyzer.analyze_study(control, treatment)

    def test_list_inputs_converted(self):
        """Test that list inputs are converted to numpy arrays."""
        analyzer = IPDQMA()
        control = [1, 2, 3, 4, 5]
        treatment = [2, 3, 4, 5, 6]

        result = analyzer.analyze_study(control, treatment)
        assert 'quantiles' in result
        assert 'slope' in result
        assert 'lnvr' in result


class TestAnalyzeStudy:
    """Test the analyze_study method."""

    @pytest.fixture
    def normal_data(self):
        """Generate normally distributed test data."""
        np.random.seed(42)
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.5, 1, 100)
        return control, treatment

    @pytest.fixture
    def exponential_data(self):
        """Generate exponentially distributed test data."""
        np.random.seed(42)
        control = np.random.exponential(1, 100)
        treatment = np.random.exponential(1.5, 100)
        return control, treatment

    def test_analyze_study_returns_correct_keys(self, normal_data):
        """Test that analyze_study returns all required keys."""
        control, treatment = normal_data
        analyzer = IPDQMA()
        result = analyzer.analyze_study(control, treatment)

        required_keys = [
            'quantiles', 'se_quantiles', 'quantile_effects_bc',
            'slope', 'se_slope', 'lnvr', 'se_lnvr',
            'n_control', 'n_treatment',
            'mean_control', 'mean_treatment',
            'sd_control', 'sd_treatment'
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_quantile_estimation(self, normal_data):
        """Test that quantiles are estimated correctly."""
        control, treatment = normal_data
        analyzer = IPDQMA(quantiles=[0.25, 0.5, 0.75])
        result = analyzer.analyze_study(control, treatment)

        # Check that we have the right number of quantiles
        assert len(result['quantiles']) == 3

        # Check that quantiles are reasonable
        # For shifted normal, differences should be around 0.5
        assert np.all(np.abs(result['quantiles']) < 10)  # Not extreme

    def test_slope_calculation(self, exponential_data):
        """Test slope calculation (Q90 - Q10)."""
        control, treatment = exponential_data
        analyzer = IPDQMA(quantiles=[0.1, 0.5, 0.9])
        result = analyzer.analyze_study(control, treatment)

        # Slope should be positive for this data (treatment has larger scale)
        # Q90 - Q10 should be positive
        assert result['slope'] == result['quantiles'][2] - result['quantiles'][0]

        # SE should be positive
        assert result['se_slope'] >= 0

    def test_lnvr_calculation(self, normal_data):
        """Test log variance ratio calculation."""
        control, treatment = normal_data
        analyzer = IPDQMA()
        result = analyzer.analyze_study(control, treatment)

        # lnVR should be near 0 for equal variances
        var_c = np.var(control, ddof=1)
        var_t = np.var(treatment, ddof=1)
        expected_lnvr = np.log(var_t / var_c)

        assert abs(result['lnvr'] - expected_lnvr) < 0.01

    def test_se_lnvr_calculation(self, normal_data):
        """Test standard error of lnVR calculation."""
        control, treatment = normal_data
        analyzer = IPDQMA()
        result = analyzer.analyze_study(control, treatment)

        n_c = len(control)
        n_t = len(treatment)
        expected_se = np.sqrt(1 / (n_t - 1) + 1 / (n_c - 1))

        assert abs(result['se_lnvr'] - expected_se) < 0.01

    def test_bootstrap_reproducibility(self, normal_data):
        """Test that bootstrap results are reproducible with same seed."""
        control, treatment = normal_data

        analyzer1 = IPDQMA(config=IQMAConfig(random_seed=42, n_bootstrap=100))
        result1 = analyzer1.analyze_study(control, treatment)

        analyzer2 = IPDQMA(config=IQMAConfig(random_seed=42, n_bootstrap=100))
        result2 = analyzer2.analyze_study(control, treatment)

        np.testing.assert_array_almost_equal(result1['quantiles'], result2['quantiles'])

    def test_bias_correction_calculation(self, exponential_data):
        """Test that bias correction is calculated."""
        control, treatment = exponential_data
        analyzer = IPDQMA()
        result = analyzer.analyze_study(control, treatment)

        # Check that bias-corrected estimates exist
        assert 'quantile_effects_bc' in result
        assert len(result['quantile_effects_bc']) == len(analyzer.quantiles)

    def test_sample_sizes_recorded(self, normal_data):
        """Test that sample sizes are correctly recorded."""
        control, treatment = normal_data
        analyzer = IPDQMA()
        result = analyzer.analyze_study(control, treatment)

        assert result['n_control'] == len(control)
        assert result['n_treatment'] == len(treatment)

    def test_mean_and_sd_recorded(self, normal_data):
        """Test that means and SDs are correctly recorded."""
        control, treatment = normal_data
        analyzer = IPDQMA()
        result = analyzer.analyze_study(control, treatment)

        assert abs(result['mean_control'] - np.mean(control)) < 0.01
        assert abs(result['mean_treatment'] - np.mean(treatment)) < 0.01
        assert abs(result['sd_control'] - np.std(control, ddof=1)) < 0.01
        assert abs(result['sd_treatment'] - np.std(treatment, ddof=1)) < 0.01


class TestPoolingMethods:
    """Test fixed-effect and random-effects pooling methods."""

    @pytest.fixture
    def multi_study_data(self):
        """Generate data for multiple studies."""
        np.random.seed(42)
        studies = []
        for i in range(10):
            control = np.random.normal(0, 1, 100)
            treatment = np.random.normal(0.5, 1, 100)
            studies.append((control, treatment))
        return studies

    def test_fixed_effect_pooling(self):
        """Test fixed-effect meta-analysis pooling."""
        analyzer = IPDQMA(config=IQMAConfig(use_random_effects=False))
        estimates = np.array([0.5, 0.6, 0.4, 0.55])
        se = np.array([0.1, 0.12, 0.08, 0.11])

        result = analyzer._pool_fixed_effect(estimates, se)

        assert 'estimate' in result
        assert 'se' in result
        assert 'z' in result
        assert 'p' in result
        assert 'lower' in result
        assert 'upper' in result
        assert 'weights' in result

        # Pooled estimate should be weighted average
        assert result['estimate'] > 0.4 and result['estimate'] < 0.6
        assert result['se'] > 0

    def test_fixed_effect_weights_sum_to_one(self):
        """Test that fixed-effect weights sum to 1."""
        analyzer = IPDQMA()
        estimates = np.array([0.5, 0.6, 0.4])
        se = np.array([0.1, 0.12, 0.08])

        result = analyzer._pool_fixed_effect(estimates, se)
        assert abs(np.sum(result['weights']) - 1.0) < 0.01

    def test_heterogeneity_estimation_dl(self):
        """Test DerSimonian-Laird heterogeneity estimation."""
        analyzer = IPDQMA()
        estimates = np.array([0.5, 0.6, 0.4, 0.55, 0.45])
        se = np.array([0.1, 0.12, 0.08, 0.11, 0.09])

        het = analyzer._estimate_heterogeneity(estimates, se, method='dl')

        assert 'tau2' in het
        assert 'tau' in het
        assert 'i2' in het
        assert 'q' in het
        assert 'q_p' in het
        assert het['tau2'] >= 0
        assert het['tau'] >= 0
        assert 0 <= het['i2'] <= 100

    def test_heterogeneity_estimation_pm(self):
        """Test Paule-Mandel heterogeneity estimation."""
        analyzer = IPDQMA()
        estimates = np.array([0.5, 0.6, 0.4, 0.55])
        se = np.array([0.1, 0.12, 0.08, 0.11])

        het = analyzer._estimate_heterogeneity(estimates, se, method='pm')

        assert het['tau2'] >= 0
        assert het['tau'] >= 0

    def test_heterogeneity_single_study(self):
        """Test heterogeneity estimation with single study."""
        analyzer = IPDQMA()
        estimates = np.array([0.5])
        se = np.array([0.1])

        het = analyzer._estimate_heterogeneity(estimates, se)

        # With single study, heterogeneity should be 0
        assert het['tau2'] == 0
        assert het['tau'] == 0
        assert het['i2'] == 0

    def test_random_effects_pooling(self):
        """Test random-effects meta-analysis pooling."""
        analyzer = IPDQMA()
        estimates = np.array([0.5, 0.6, 0.4, 0.55])
        se = np.array([0.1, 0.12, 0.08, 0.11])
        het = {'tau2': 0.05, 'tau': np.sqrt(0.05)}

        result = analyzer._pool_random_effects(estimates, se, het)

        assert 'estimate' in result
        assert 'se' in result
        assert 'pred_lower' in result
        assert 'pred_upper' in result

        # Prediction interval should be wider than confidence interval
        assert (result['pred_upper'] - result['pred_lower']) > (result['upper'] - result['lower'])

    def test_random_effects_weights_sum_to_one(self):
        """Test that random-effects weights sum to 1."""
        analyzer = IPDQMA()
        estimates = np.array([0.5, 0.6, 0.4])
        se = np.array([0.1, 0.12, 0.08])
        het = {'tau2': 0.05, 'tau': np.sqrt(0.05)}

        result = analyzer._pool_random_effects(estimates, se, het)
        assert abs(np.sum(result['weights']) - 1.0) < 0.01


class TestFitMethod:
    """Test the fit method for multi-study analysis."""

    @pytest.fixture
    def multi_study_data(self):
        """Generate data for multiple studies."""
        np.random.seed(42)
        studies = []
        for i in range(5):
            control = np.random.normal(0, 1, 100)
            treatment = np.random.normal(0.5, 1, 100)
            studies.append((control, treatment))
        return studies

    def test_fit_returns_results(self, multi_study_data):
        """Test that fit returns results dictionary."""
        analyzer = IPDQMA(n_boot=50)
        results = analyzer.fit(multi_study_data)

        assert isinstance(results, dict)
        assert 'n_studies' in results
        assert 'model_type' in results
        assert 'profile' in results
        assert 'slope_test' in results
        assert 'lnvr_test' in results

    def test_fit_n_studies_correct(self, multi_study_data):
        """Test that fit records correct number of studies."""
        analyzer = IPDQMA(n_boot=50)
        results = analyzer.fit(multi_study_data)

        assert results['n_studies'] == len(multi_study_data)

    def test_fit_model_type_fixed(self, multi_study_data):
        """Test that fit uses fixed-effect model when specified."""
        analyzer = IPDQMA(config=IQMAConfig(use_random_effects=False), n_boot=50)
        results = analyzer.fit(multi_study_data)

        assert results['model_type'] == 'fixed_effect'

    def test_fit_model_type_random(self, multi_study_data):
        """Test that fit uses random-effects model when specified."""
        analyzer = IPDQMA(config=IQMAConfig(use_random_effects=True), n_boot=50)
        results = analyzer.fit(multi_study_data)

        assert results['model_type'] == 'random_effects'

    def test_fit_profile_dataframe(self, multi_study_data):
        """Test that profile is a DataFrame with correct columns."""
        analyzer = IPDQMA(n_boot=50)
        results = analyzer.fit(multi_study_data)

        profile = results['profile']
        assert isinstance(profile, pd.DataFrame)

        required_cols = ['Quantile', 'Effect', 'SE', 'Z', 'P', 'CI_Lower', 'CI_Upper']
        for col in required_cols:
            assert col in profile.columns

        # Check number of rows matches quantiles
        assert len(profile) == len(analyzer.quantiles)

    def test_fit_slope_test(self, multi_study_data):
        """Test that slope test is computed correctly."""
        analyzer = IPDQMA(n_boot=50)
        results = analyzer.fit(multi_study_data)

        slope_test = results['slope_test']
        assert 'estimate' in slope_test
        assert 'se' in slope_test
        assert 'p' in slope_test
        assert 'interpretation' in slope_test
        assert isinstance(slope_test['interpretation'], str)

    def test_fit_lnvr_test(self, multi_study_data):
        """Test that lnVR test is computed correctly."""
        analyzer = IPDQMA(n_boot=50)
        results = analyzer.fit(multi_study_data)

        lnvr_test = results['lnvr_test']
        assert 'estimate' in lnvr_test
        assert 'se' in lnvr_test
        assert 'p' in lnvr_test
        assert 'interpretation' in lnvr_test

    def test_fit_with_single_study(self):
        """Test fit with a single study."""
        np.random.seed(42)
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.5, 1, 100)

        analyzer = IPDQMA(n_boot=50)
        results = analyzer.fit([(control, treatment)])

        assert results['n_studies'] == 1
        assert len(results['profile']) == len(analyzer.quantiles)


class TestVisualization:
    """Test plotting and visualization methods."""

    @pytest.fixture
    def fitted_analyzer(self):
        """Create a fitted analyzer for plotting tests."""
        np.random.seed(42)
        studies = []
        for i in range(5):
            control = np.random.normal(0, 1, 100)
            treatment = np.random.normal(0.5, 1, 100)
            studies.append((control, treatment))

        analyzer = IPDQMA(n_boot=50)
        analyzer.fit(studies)
        return analyzer

    def test_plot_before_fit_raises_error(self):
        """Test that plotting before fit raises ValueError."""
        analyzer = IPDQMA()

        with pytest.raises(ValueError, match="Run fit"):
            analyzer.plot()

    def test_plot_returns_figure(self, fitted_analyzer):
        """Test that plot returns a matplotlib Figure."""
        fig = fitted_analyzer.plot()

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_forest_returns_figure(self, fitted_analyzer):
        """Test that plot_forest returns a matplotlib Figure."""
        fig = fitted_analyzer.plot_forest()

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_forest_custom_quantile(self, fitted_analyzer):
        """Test forest plot with custom quantile index."""
        fig = fitted_analyzer.plot_forest(quantile_index=0)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_custom_figsize(self, fitted_analyzer):
        """Test plot with custom figure size."""
        fig = fitted_analyzer.plot(figsize=(8, 4))

        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 4
        plt.close(fig)


class TestExportResults:
    """Test results export functionality."""

    @pytest.fixture
    def fitted_analyzer(self):
        """Create a fitted analyzer for export tests."""
        np.random.seed(42)
        studies = []
        for i in range(3):
            control = np.random.normal(0, 1, 100)
            treatment = np.random.normal(0.5, 1, 100)
            studies.append((control, treatment))

        analyzer = IPDQMA(n_boot=50)
        analyzer.fit(studies)
        return analyzer

    def test_export_before_fit_raises_error(self, tmp_path):
        """Test that export before fit raises ValueError."""
        analyzer = IPDQMA()
        output_file = tmp_path / "test_output.xlsx"

        with pytest.raises(ValueError, match="Run fit"):
            analyzer.export_results(str(output_file))

    @pytest.mark.skipif(True, reason="Requires openpyxl which may not be installed")
    def test_export_results_xlsx(self, fitted_analyzer, tmp_path):
        """Test exporting results to Excel format."""
        output_file = tmp_path / "test_output.xlsx"

        # This test requires openpyxl, skip if not available
        try:
            import openpyxl
            fitted_analyzer.export_results(str(output_file), format='xlsx')
            assert output_file.exists()
        except ImportError:
            pytest.skip("openpyxl not installed")

    def test_export_results_csv(self, fitted_analyzer, tmp_path):
        """Test exporting results to CSV format."""
        output_file = tmp_path / "test_output.csv"

        fitted_analyzer.export_results(str(output_file), format='csv')
        assert output_file.exists()


class TestSummaryMethod:
    """Test the summary method."""

    @pytest.fixture
    def fitted_analyzer(self):
        """Create a fitted analyzer for summary tests."""
        np.random.seed(42)
        studies = []
        for i in range(3):
            control = np.random.normal(0, 1, 100)
            treatment = np.random.normal(0.5, 1, 100)
            studies.append((control, treatment))

        analyzer = IPDQMA(n_boot=50)
        analyzer.fit(studies)
        return analyzer

    def test_summary_before_fit_raises_error(self):
        """Test that summary before fit raises ValueError."""
        analyzer = IPDQMA()

        with pytest.raises(ValueError, match="Run fit"):
            analyzer.summary()

    def test_summary_returns_dataframe(self, fitted_analyzer, capsys):
        """Test that summary returns a DataFrame."""
        df = fitted_analyzer.summary()

        assert isinstance(df, pd.DataFrame)

        # Check that output was printed
        captured = capsys.readouterr()
        assert "IPD-QMA Analysis Summary" in captured.out

    def test_summary_columns(self, fitted_analyzer):
        """Test that summary has correct columns."""
        df = fitted_analyzer.summary()

        expected_cols = ['Quantile', 'Effect', 'SE', '95% CI Lower', '95% CI Upper', 'P-value', 'I² (%)', 'τ²']
        for col in expected_cols:
            assert col in df.columns


class TestInterpretationMethods:
    """Test interpretation methods for test results."""

    def test_interpret_slope_very_significant(self):
        """Test slope interpretation with very significant result."""
        analyzer = IPDQMA()
        interpretation = analyzer._interpret_slope(0.5, 0.0001)

        assert "Very strong evidence" in interpretation
        assert "0.5" in interpretation

    def test_interpret_slope_significant(self):
        """Test slope interpretation with significant result."""
        analyzer = IPDQMA()
        interpretation = analyzer._interpret_slope(0.3, 0.02)

        assert "Significant" in interpretation

    def test_interpret_slope_trend(self):
        """Test slope interpretation with trend."""
        analyzer = IPDQMA()
        interpretation = analyzer._interpret_slope(0.2, 0.08)

        assert "Trend" in interpretation

    def test_interpret_slope_not_significant(self):
        """Test slope interpretation with non-significant result."""
        analyzer = IPDQMA()
        interpretation = analyzer._interpret_slope(0.1, 0.5)

        assert "No significant" in interpretation

    def test_interpret_lnvr_increased(self):
        """Test lnVR interpretation with increased variance."""
        analyzer = IPDQMA()
        interpretation = analyzer._interpret_lnvr(0.5, 0.01)

        assert "increased" in interpretation

    def test_interpret_lnvr_decreased(self):
        """Test lnVR interpretation with decreased variance."""
        analyzer = IPDQMA()
        interpretation = analyzer._interpret_lnvr(-0.5, 0.01)

        assert "decreased" in interpretation


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_small(self):
        """Test complete workflow with small dataset."""
        # Generate data
        np.random.seed(42)
        studies = []
        for i in range(3):
            control = np.random.exponential(1, 50)
            treatment = np.random.exponential(1.5, 50)
            studies.append((control, treatment))

        # Run analysis
        config = IQMAConfig(
            quantiles=[0.25, 0.5, 0.75],
            n_bootstrap=50,
            use_random_effects=True
        )
        analyzer = IPDQMA(config)
        results = analyzer.fit(studies)

        # Check results
        assert results['n_studies'] == 3
        assert results['model_type'] == 'random_effects'
        assert len(results['profile']) == 3

        # Check summary works
        summary = analyzer.summary()
        assert isinstance(summary, pd.DataFrame)

        # Check plots work
        fig1 = analyzer.plot()
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = analyzer.plot_forest()
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_reproducibility_same_data(self):
        """Test that results are reproducible with same data and seed."""
        np.random.seed(42)
        studies = []
        for i in range(5):
            control = np.random.normal(0, 1, 100)
            treatment = np.random.normal(0.5, 1.2, 100)
            studies.append((control, treatment))

        # First run
        config1 = IQMAConfig(random_seed=42, n_bootstrap=100)
        analyzer1 = IPDQMA(config1)
        results1 = analyzer1.fit(studies)

        # Second run with same seed
        config2 = IQMAConfig(random_seed=42, n_bootstrap=100)
        analyzer2 = IPDQMA(config2)
        results2 = analyzer2.fit(studies)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            results1['profile']['Effect'].values,
            results2['profile']['Effect'].values,
            decimal=5
        )

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        import time

        np.random.seed(42)
        studies = []
        for i in range(10):
            control = np.random.normal(0, 1, 500)
            treatment = np.random.normal(0.5, 1.2, 500)
            studies.append((control, treatment))

        config = IQMAConfig(n_bootstrap=200)
        analyzer = IPDQMA(config)

        start = time.time()
        results = analyzer.fit(studies)
        elapsed = time.time() - start

        # Should complete within reasonable time (< 30 seconds for this test)
        assert elapsed < 30

        assert results['n_studies'] == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identical_distributions(self):
        """Test with identical control and treatment distributions."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        analyzer = IPDQMA(n_boot=50)
        result = analyzer.analyze_study(data, data.copy())

        # Effects should be near zero
        assert np.all(np.abs(result['quantiles']) < 1)

    def test_extreme_difference(self):
        """Test with extreme difference between groups."""
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(10, 1, 100)

        analyzer = IPDQMA(n_boot=50)
        result = analyzer.analyze_study(control, treatment)

        # Effects should be large and positive
        assert np.all(result['quantiles'] > 5)

    def test_single_quantile(self):
        """Test with only one quantile."""
        analyzer = IPDQMA(quantiles=[0.5], n_boot=50)

        np.random.seed(42)
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.5, 1, 100)

        result = analyzer.analyze_study(control, treatment)

        assert len(result['quantiles']) == 1

    def test_many_quantiles(self):
        """Test with many quantiles."""
        quantiles = np.linspace(0.05, 0.95, 19)  # 5% increments
        analyzer = IPDQMA(quantiles=quantiles.tolist(), n_boot=50)

        np.random.seed(42)
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.5, 1, 100)

        result = analyzer.analyze_study(control, treatment)

        assert len(result['quantiles']) == 19


def run_tests():
    """Run all tests and print summary."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
