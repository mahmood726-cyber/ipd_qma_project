"""
IPD-QMA Advanced: Advanced Statistical Methods for Quantile Meta-Analysis

This module extends the base IPD-QMA with advanced statistical techniques:
- Publication bias assessment (funnel plots, Egger regression for quantiles)
- Subgroup analysis by covariates
- Meta-regression for quantile effects
- Cumulative meta-analysis
- Leave-one-out sensitivity analysis
- Trim-and-fill method

Author: IPD-QMA Development Team
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
import warnings

# Import base IPDQMA
from ipd_qma import IPDQMA, IQMAConfig


class IPDQMAAdvanced(IPDQMA):
    """
    Extended IPD-QMA with advanced statistical methods.

    Inherits all functionality from IPDQMA and adds:
    - Publication bias assessment
    - Subgroup analysis
    - Meta-regression
    - Cumulative meta-analysis
    - Sensitivity analysis
    """

    def __init__(self, config: Optional[IQMAConfig] = None, **kwargs):
        """Initialize advanced analyzer."""
        super().__init__(config, **kwargs)
        self._publication_bias_results = {}
        self._subgroup_results = {}
        self._metaregression_results = {}
        self._cumulative_results = {}
        self._sensitivity_results = {}

    # ========================================================================
    # PUBLICATION BIAS ASSESSMENT
    # ========================================================================

    def assess_publication_bias(
        self,
        quantile_index: int = -1,
        method: str = 'both'
    ) -> Dict:
        """
        Assess publication bias using funnel plot and Egger regression.

        Parameters
        ----------
        quantile_index : int
            Index of quantile to assess (default: median)
        method : str
            'funnel', 'egger', or 'both'

        Returns
        -------
        dict
            Publication bias assessment results
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2

        results = {
            'quantile': self.quantiles[q_idx],
            'quantile_index': q_idx
        }

        # Extract study effects and SEs for this quantile
        effects = np.array([s['quantiles'][q_idx] for s in self._study_results])
        se = np.array([s['se_quantiles'][q_idx] for s in self._study_results])
        n_studies = len(effects)

        if n_studies < 3:
            warnings.warn("Too few studies for publication bias assessment")
            results['error'] = "Too few studies"
            return results

        # Funnel plot asymmetry test
        if method in ['funnel', 'both']:
            results['funnel'] = self._funnel_plot_test(effects, se)

        # Egger regression
        if method in ['egger', 'both']:
            results['egger'] = self._egger_regression(effects, se)

        self._publication_bias_results[q_idx] = results
        return results

    def _funnel_plot_test(
        self,
        effects: np.ndarray,
        se: np.ndarray
    ) -> Dict:
        """
        Test funnel plot asymmetry using rank correlation.

        Parameters
        ----------
        effects : array-like
            Study effects
        se : array-like
            Standard errors

        Returns
        -------
        dict
            Funnel plot test results
        """
        # Begg's rank correlation test
        # Correlation between standardized effect and variance
        n = len(effects)
        k = se ** -2  # Precision
        z = effects / se  # Standardized effects

        # Rank correlation
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(k, z)

        return {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': self._interpret_funnel_test(corr, p_value)
        }

    def _egger_regression(
        self,
        effects: np.ndarray,
        se: np.ndarray
    ) -> Dict:
        """
        Egger regression test for funnel plot asymmetry.

        Parameters
        ----------
        effects : array-like
            Study effects
        se : array-like
            Standard errors

        Returns
        -------
        dict
            Egger regression results
        """
        # Standardized effect vs precision
        precision = 1 / se
        std_effect = effects / se

        # Regression: std_effect = beta0 + beta1 * precision
        X = np.column_stack([np.ones_like(precision), precision])
        y = std_effect

        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Calculate residuals, t-statistic, p-value
        residuals = y - X @ beta
        n = len(effects)
        mse = np.sum(residuals ** 2) / (n - 2)
        se_beta = np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X)))
        t_stat = beta / se_beta
        p_values = 2 * (1 - stats.t.sf(np.abs(t_stat), n - 2))

        # Intercept (beta0) test - significance indicates asymmetry
        intercept_p = p_values[0]

        return {
            'intercept': beta[0],
            'intercept_se': se_beta[0],
            'intercept_p': intercept_p,
            'slope': beta[1],
            'slope_se': se_beta[1],
            'slope_p': p_values[1],
            'asymmetry': intercept_p < 0.05,
            'interpretation': self._interpret_egger_test(beta[0], intercept_p)
        }

    def _interpret_funnel_test(self, corr: float, p_value: float) -> str:
        """Interpret funnel plot test result."""
        if p_value >= 0.05:
            return f"No significant funnel asymmetry (p={p_value:.3f})"
        else:
            direction = "negative" if corr < 0 else "positive"
            return f"Significant funnel asymmetry detected (p={p_value:.3f}, correlation={direction})"

    def _interpret_egger_test(self, intercept: float, p_value: float) -> str:
        """Interpret Egger regression result."""
        if p_value >= 0.05:
            return f"No significant asymmetry (Egger test p={p_value:.3f})"
        else:
            direction = "smaller studies show larger effects" if intercept > 0 else "smaller studies show smaller effects"
            return f"Significant publication bias (p={p_value:.3f}): {direction}"

    def plot_funnel(
        self,
        quantile_index: int = -1,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Create funnel plot for publication bias assessment.

        Parameters
        ----------
        quantile_index : int
            Index of quantile to plot
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2

        effects = np.array([s['quantiles'][q_idx] for s in self._study_results])
        se = np.array([s['se_quantiles'][q_idx] for s in self._study_results])

        # Pooled effect
        pooled = self.results['profile'].iloc[q_idx]['Effect']

        fig, ax = plt.subplots(figsize=figsize)

        # Funnel plot: effect vs precision (1/SE)
        precision = 1 / se

        # Reference lines
        ax.axvline(pooled, color='red', linestyle='--', label='Pooled effect')
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)

        # 95% CI limits
        ci_upper = pooled + 1.96 * np.mean(se)
        ci_lower = pooled - 1.96 * np.mean(se)
        ax.axvline(ci_upper, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(ci_lower, color='gray', linestyle=':', alpha=0.5)

        # Study points
        ax.scatter(precision, effects, s=50, alpha=0.6, edgecolors='black')

        # 95% pseudo-confidence region
        y_range = np.linspace(effects.min(), effects.max(), 100)
        x_upper = (pooled + 1.96 / precision) if len(precision) > 0 else pooled
        x_lower = (pooled - 1.96 / precision) if len(precision) > 0 else pooled

        ax.set_xlabel('Precision (1/SE)')
        ax.set_ylabel(f'Effect at {self.quantiles[q_idx]:.0%} quantile')
        ax.set_title(f'Funnel Plot - {self.quantiles[q_idx]:.0%} Quantile')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # ========================================================================
    # SUBGROUP ANALYSIS
    # ========================================================================

    def subgroup_analysis(
        self,
        subgroup_labels: List[str],
        quantile_index: int = -1
    ) -> Dict:
        """
        Perform subgroup analysis by categorical variable.

        Parameters
        ----------
        subgroup_labels : list of str
            Subgroup label for each study (e.g., ['A', 'B', 'A', 'C', ...])
        quantile_index : int
            Index of quantile to analyze

        Returns
        -------
        dict
            Subgroup analysis results
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        if len(subgroup_labels) != len(self._study_results):
            raise ValueError("Number of labels must match number of studies")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2

        # Group studies
        subgroups = {}
        for i, label in enumerate(subgroup_labels):
            if label not in subgroups:
                subgroups[label] = []
            subgroups[label].append(i)

        # Analyze each subgroup
        results = {
            'quantile': self.quantiles[q_idx],
            'subgroups': {},
            'between_group_test': {}
        }

        subgroup_effects = []
        subgroup_se = []
        subgroup_names = []

        for label, study_indices in subgroups.items():
            # Extract effects for this subgroup
            effects = np.array([self._study_results[i]['quantiles'][q_idx]
                              for i in study_indices])
            se = np.array([self._study_results[i]['se_quantiles'][q_idx]
                          for i in study_indices])

            # Pool within subgroup
            pooled = self._pool_fixed_effect(effects, se)

            results['subgroups'][label] = {
                'n_studies': len(study_indices),
                'effect': pooled['estimate'],
                'se': pooled['se'],
                'ci_lower': pooled['lower'],
                'ci_upper': pooled['upper'],
                'p_value': pooled['p'],
                'study_indices': study_indices
            }

            subgroup_effects.append(pooled['estimate'])
            subgroup_se.append(pooled['se'])
            subgroup_names.append(label)

        # Between-group test
        results['between_group_test'] = self._between_group_test(
            subgroup_effects, subgroup_se, subgroup_names
        )

        self._subgroup_results[q_idx] = results
        return results

    def _between_group_test(
        self,
        effects: List[float],
        se: List[float],
        names: List[str]
    ) -> Dict:
        """
        Test for differences between subgroups.

        Uses Q-test for heterogeneity between subgroups.
        """
        k = len(effects)

        if k < 2:
            return {'error': 'Need at least 2 subgroups'}

        # Weighted mean
        weights = np.array([1 / (s ** 2) for s in se])
        pooled = np.sum(np.array(effects) * weights) / np.sum(weights)

        # Q-statistic (between groups)
        q_between = np.sum(weights * (np.array(effects) - pooled) ** 2)
        df = k - 1
        p_value = 1 - stats.chi2.cdf(q_between, df)

        return {
            'q_statistic': q_between,
            'df': df,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': self._interpret_subgroup_test(p_value, df)
        }

    def _interpret_subgroup_test(self, p_value: float, df: int) -> str:
        """Interpret subgroup test result."""
        if p_value >= 0.05:
            return f"No significant difference between subgroups (Q={df} df, p={p_value:.3f})"
        else:
            return f"Significant differences between subgroups (Q={df} df, p={p_value:.3f})"

    def plot_subgroup_forest(
        self,
        subgroup_results: Optional[Dict] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create forest plot comparing subgroups.

        Parameters
        ----------
        subgroup_results : dict, optional
            Results from subgroup_analysis()
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if subgroup_results is None:
            if not self._subgroup_results:
                raise ValueError("Run subgroup_analysis() first")
            # Use most recent result
            subgroup_results = list(self._subgroup_results.values())[0]

        subgroups = subgroup_results['subgroups']
        names = list(subgroups.keys())

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(names))
        effects = [subgroups[n]['effect'] for n in names]
        se = [subgroups[n]['se'] for n in names]
        ci_lower = [subgroups[n]['ci_lower'] for n in names]
        ci_upper = [subgroups[n]['ci_upper'] for n in names]

        # Forest plot
        ax.errorbar(
            effects,
            y_pos,
            xerr=[np.array(effects) - np.array(ci_lower),
                  np.array(ci_upper) - np.array(effects)],
            fmt='o',
            capsize=5,
            markersize=8,
            color='#2E86C1'
        )

        # Reference line
        ax.axvline(0, color='gray', linestyle='-', linewidth=1)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{n} (n={subgroups[n]['n_studies']})" for n in names])
        ax.set_xlabel('Effect Size')
        ax.set_title('Subgroup Analysis')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    # ========================================================================
    # META-REGRESSION
    # ========================================================================

    def meta_regression(
        self,
        covariates: np.ndarray,
        quantile_index: int = -1,
        method: str = 'fixed'
    ) -> Dict:
        """
        Meta-regression of quantile effects on study-level covariates.

        Parameters
        ----------
        covariates : array-like
            Study-level covariates (n_studies x n_covariates)
        quantile_index : int
            Index of quantile to analyze
        method : str
            'fixed' or 'random' effects model

        Returns
        -------
        dict
            Meta-regression results
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2

        effects = np.array([s['quantiles'][q_idx] for s in self._study_results])
        se = np.array([s['se_quantiles'][q_idx] for s in self._study_results])

        covariates = np.asarray(covariates)

        if len(covariates) != len(effects):
            raise ValueError("Number of covariate rows must match number of studies")

        # Weighted least squares
        weights = 1 / (se ** 2)
        W = np.diag(weights)

        # Add intercept
        X = np.column_stack([np.ones(len(effects)), covariates])
        y = effects

        # WLS estimation
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta = np.linalg.solve(XtWX, XtWy)

        # Standard errors
        mse = np.sum(weights * (y - X @ beta) ** 2) / (len(effects) - X.shape[1])
        var_beta = mse * np.linalg.inv(XtWX)
        se_beta = np.sqrt(np.diag(var_beta))

        # t-statistics and p-values
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.sf(np.abs(t_stats), len(effects) - X.shape[1]))

        # R-squared
        y_pred = X @ beta
        ss_total = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        ss_residual = np.sum(weights * (y - y_pred) ** 2)
        r_squared = 1 - ss_residual / ss_total

        results = {
            'quantile': self.quantiles[q_idx],
            'coefficients': beta,
            'se': se_beta,
            't_stats': t_stats,
            'p_values': p_values,
            'r_squared': r_squared,
            'n_studies': len(effects),
            'n_covariates': X.shape[1] - 1
        }

        self._metaregression_results[q_idx] = results
        return results

    def plot_meta_regression(
        self,
        covariate: np.ndarray,
        covariate_name: str = 'Covariate',
        quantile_index: int = -1,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot meta-regression results.

        Parameters
        ----------
        covariate : array-like
            Single covariate values
        covariate_name : str
            Name for covariate axis
        quantile_index : int
            Index of quantile to plot
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2

        effects = np.array([s['quantiles'][q_idx] for s in self._study_results])
        se = np.array([s['se_quantiles'][q_idx] for s in self._study_results])

        # Fit regression
        reg_result = self.meta_regression(
            covariate.reshape(-1, 1) if covariate.ndim == 1 else covariate,
            quantile_index=q_idx
        )

        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot (point size by precision)
        precision = 1 / se
        ax.scatter(covariate, effects, s=precision * 50, alpha=0.6, edgecolors='black')

        # Regression line
        if covariate.ndim == 1:
            x_range = np.linspace(covariate.min(), covariate.max(), 100)
            y_pred = reg_result['coefficients'][0] + reg_result['coefficients'][1] * x_range
            ax.plot(x_range, y_pred, 'r-', linewidth=2, label='Regression line')

        # Error bars
        ax.errorbar(covariate, effects, yerr=1.96 * se, fmt='none', alpha=0.3)

        ax.set_xlabel(covariate_name)
        ax.set_ylabel(f'Effect at {self.quantiles[q_idx]:.0%} quantile')
        ax.set_title('Meta-Regression')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # ========================================================================
    # CUMULATIVE META-ANALYSIS
    # ========================================================================

    def cumulative_meta_analysis(
        self,
        quantile_index: int = -1,
        order: Optional[List[int]] = None
    ) -> Dict:
        """
        Perform cumulative meta-analysis (studies added sequentially).

        Parameters
        ----------
        quantile_index : int
            Index of quantile to analyze
        order : list of int, optional
            Order to add studies (default: original order)

        Returns
        -------
        dict
            Cumulative meta-analysis results
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2

        if order is None:
            order = list(range(len(self._study_results)))

        cumulative_results = []
        current_studies = []

        for i, study_idx in enumerate(order):
            current_studies.append(study_idx)

            # Get effects for accumulated studies
            effects = np.array([self._study_results[idx]['quantiles'][q_idx]
                              for idx in current_studies])
            se = np.array([self._study_results[idx]['se_quantiles'][q_idx]
                          for idx in current_studies])

            # Pool
            if self.config.use_random_effects:
                het = self._estimate_heterogeneity(effects, se)
                pooled = self._pool_random_effects(effects, se, het)
            else:
                pooled = self._pool_fixed_effect(effects, se)

            cumulative_results.append({
                'step': i + 1,
                'n_studies': len(current_studies),
                'effect': pooled['estimate'],
                'se': pooled['se'],
                'ci_lower': pooled['lower'],
                'ci_upper': pooled['upper'],
                'p_value': pooled['p'],
                'added_study': study_idx
            })

        results = {
            'quantile': self.quantiles[q_idx],
            'order': order,
            'cumulative': cumulative_results
        }

        self._cumulative_results[q_idx] = results
        return results

    def plot_cumulative(
        self,
        cumulative_results: Optional[Dict] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot cumulative meta-analysis results.

        Parameters
        ----------
        cumulative_results : dict, optional
            Results from cumulative_meta_analysis()
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if cumulative_results is None:
            if not self._cumulative_results:
                raise ValueError("Run cumulative_meta_analysis() first")
            cumulative_results = list(self._cumulative_results.values())[0]

        cumulative = cumulative_results['cumulative']

        fig, ax = plt.subplots(figsize=figsize)

        steps = [c['step'] for c in cumulative]
        effects = [c['effect'] for c in cumulative]
        ci_lower = [c['ci_lower'] for c in cumulative]
        ci_upper = [c['ci_upper'] for c in cumulative]

        ax.plot(steps, effects, 'o-', color='#2E86C1', linewidth=2, markersize=6)
        ax.fill_between(steps, ci_lower, ci_upper, alpha=0.3, color='#2E86C1')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        ax.set_xlabel('Number of studies')
        ax.set_ylabel('Pooled effect size')
        ax.set_title('Cumulative Meta-Analysis')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # ========================================================================
    # LEAVE-ONE-OUT SENSITIVITY ANALYSIS
    # ========================================================================

    def leave_one_out(
        self,
        quantile_index: int = -1
    ) -> Dict:
        """
        Perform leave-one-out sensitivity analysis.

        Parameters
        ----------
        quantile_index : int
            Index of quantile to analyze

        Returns
        -------
        dict
            Leave-one-out analysis results
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2

        results = []
        n_studies = len(self._study_results)

        for leave_out in range(n_studies):
            # Exclude one study
            include_idx = [i for i in range(n_studies) if i != leave_out]

            effects = np.array([self._study_results[i]['quantiles'][q_idx]
                              for i in include_idx])
            se = np.array([self._study_results[i]['se_quantiles'][q_idx]
                          for i in include_idx])

            # Pool
            if self.config.use_random_effects:
                het = self._estimate_heterogeneity(effects, se)
                pooled = self._pool_random_effects(effects, se, het)
            else:
                pooled = self._pool_fixed_effect(effects, se)

            results.append({
                'left_out': leave_out,
                'effect': pooled['estimate'],
                'se': pooled['se'],
                'ci_lower': pooled['lower'],
                'ci_upper': pooled['upper'],
                'p_value': pooled['p']
            })

        # Original result
        original = self.results['profile'].iloc[q_idx]

        summary = {
            'quantile': self.quantiles[q_idx],
            'original': {
                'effect': original['Effect'],
                'ci_lower': original['CI_Lower'],
                'ci_upper': original['CI_Upper']
            },
            'leave_one_out': results
        }

        self._sensitivity_results[q_idx] = summary
        return summary

    def plot_leave_one_out(
        self,
        sensitivity_results: Optional[Dict] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot leave-one-out sensitivity analysis.

        Parameters
        ----------
        sensitivity_results : dict, optional
            Results from leave_one_out()
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if sensitivity_results is None:
            if not self._sensitivity_results:
                raise ValueError("Run leave_one_out() first")
            sensitivity_results = list(self._sensitivity_results.values())[0]

        leave_out = sensitivity_results['leave_one_out']
        original = sensitivity_results['original']

        fig, ax = plt.subplots(figsize=figsize)

        left_out_idx = [lo['left_out'] for lo in leave_out]
        effects = [lo['effect'] for lo in leave_out]
        ci_lower = [lo['ci_lower'] for lo in leave_out]
        ci_upper = [lo['ci_upper'] for lo in leave_out]

        # Plot leave-one-out results
        ax.errorbar(
            left_out_idx,
            effects,
            yerr=[np.array(effects) - np.array(ci_lower),
                  np.array(ci_upper) - np.array(effects)],
            fmt='o',
            capsize=5,
            color='#2E86C1',
            label='Leave-one-out'
        )

        # Original result
        ax.axhline(
            original['effect'],
            color='red',
            linestyle='--',
            linewidth=2,
            label='Original (all studies)'
        )

        # Original CI
        ax.axhline(
            original['ci_lower'],
            color='red',
            linestyle=':',
            alpha=0.5
        )
        ax.axhline(
            original['ci_upper'],
            color='red',
            linestyle=':',
            alpha=0.5
        )

        ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Study left out')
        ax.set_ylabel('Effect size')
        ax.set_title('Leave-One-Out Sensitivity Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # ========================================================================
    # TRIM-AND-FILL METHOD
    # ========================================================================

    def trim_and_fill(
        self,
        quantile_index: int = -1,
        estimator: str = 'L0',
        max_iter: int = 100
    ) -> Dict:
        """
        Trim-and-fill method for publication bias adjustment.

        Parameters
        ----------
        quantile_index : int
            Index of quantile to analyze
        estimator : str
            Estimator type ('L0', 'R0', 'Q0')
        max_iter : int
            Maximum iterations

        Returns
        -------
        dict
            Trim-and-fill results
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2

        effects = np.array([s['quantiles'][q_idx] for s in self._study_results])
        se = np.array([s['se_quantiles'][q_idx] for s in self._study_results])

        # Simple trim-and-fill implementation
        # Iteratively remove studies on one side and re-estimate

        # Initial estimate
        if self.config.use_random_effects:
            het = self._estimate_heterogeneity(effects, se)
            pooled = self._pool_random_effects(effects, se, het)
        else:
            pooled = self._pool_fixed_effect(effects, se)

        current_pooled = pooled['estimate']
        current_effects = effects.copy()
        current_se = se.copy()
        trimmed = []
        iteration = 0

        while iteration < max_iter:
            iteration += 1

            # Identify studies furthest from pooled estimate
            deviations = current_effects - current_pooled
            max_dev_idx = np.argmax(np.abs(deviations))

            # If max deviation is small, stop
            if np.abs(deviations[max_dev_idx]) < np.mean(current_se):
                break

            # Trim the most deviant study
            trimmed.append({
                'study_idx': int(np.where(np.abs(effects - current_pooled) ==
                                        np.abs(effects - current_pooled).max())[0][0]),
                'effect': float(current_effects[max_dev_idx]),
                'se': float(current_se[max_dev_idx])
            })

            # Remove from current set
            mask = np.arange(len(current_effects)) != max_dev_idx
            current_effects = current_effects[mask]
            current_se = current_se[mask]

            if len(current_effects) < 3:
                break

            # Re-estimate
            if self.config.use_random_effects:
                het = self._estimate_heterogeneity(current_effects, current_se)
                new_pooled = self._pool_random_effects(current_effects, current_se, het)
            else:
                new_pooled = self._pool_fixed_effect(current_effects, current_se)

            current_pooled = new_pooled['estimate']

        # Add back imputed studies (mirror around adjusted estimate)
        n_trimmed = len(trimmed)
        imputed = []

        for i in range(n_trimmed):
            # Create imputed study as mirror of trimmed study
            orig = trimmed[n_trimmed - 1 - i]
            deviation = orig['effect'] - current_pooled
            imputed_effect = current_pooled - deviation

            imputed.append({
                'effect': imputed_effect,
                'se': orig['se'],
                'imputed': True
            })

        # Final estimate with imputed studies
        all_effects = np.concatenate([
            [s['quantiles'][q_idx] for s in self._study_results],
            [imp['effect'] for imp in imputed]
        ])
        all_se = np.concatenate([
            [s['se_quantiles'][q_idx] for s in self._study_results],
            [imp['se'] for imp in imputed]
        ])

        if self.config.use_random_effects:
            het_final = self._estimate_heterogeneity(all_effects, all_se)
            pooled_final = self._pool_random_effects(all_effects, all_se, het_final)
        else:
            pooled_final = self._pool_fixed_effect(all_effects, all_se)

        results = {
            'quantile': self.quantiles[q_idx],
            'original_effect': pooled['estimate'],
            'original_ci_lower': pooled['lower'],
            'original_ci_upper': pooled['upper'],
            'adjusted_effect': pooled_final['estimate'],
            'adjusted_ci_lower': pooled_final['lower'],
            'adjusted_ci_upper': pooled_final['upper'],
            'n_trimmed': n_trimmed,
            'n_imputed': n_trimmed,
            'trimmed_studies': trimmed,
            'imputed_studies': imputed
        }

        return results


# Convenience function for running advanced analysis
def run_advanced_analysis(
    studies_data: List[Tuple],
    config: Optional[IQMAConfig] = None
) -> IPDQMAAdvanced:
    """
    Run advanced IPD-QMA analysis.

    Parameters
    ----------
    studies_data : list of tuples
        Study data (control, treatment) pairs
    config : IQMAConfig, optional
        Analysis configuration

    Returns
    -------
    IPDQMAAdvanced
        Fitted advanced analyzer with all results
    """
    analyzer = IPDQMAAdvanced(config)
    analyzer.fit(studies_data)
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("IPD-QMA Advanced Module")
    print("=" * 50)

    # Generate example data
    np.random.seed(42)
    studies = []
    for i in range(10):
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.5, 1.2, 100)
        studies.append((control, treatment))

    # Run analysis
    config = IQMAConfig(n_bootstrap=500, use_random_effects=True)
    analyzer = IPDQMAAdvanced(config)
    analyzer.fit(studies)

    print(f"\nAnalyzed {len(studies)} studies")

    # Publication bias
    print("\n[1] Publication Bias Assessment")
    pb = analyzer.assess_publication_bias()
    print(f"  Egger test p-value: {pb['egger']['intercept_p']:.3f}")
    print(f"  {pb['egger']['interpretation']}")

    # Leave-one-out
    print("\n[2] Leave-One-Out Sensitivity")
    loo = analyzer.leave_one_out()
    print(f"  Original effect: {loo['original']['effect']:.3f}")

    # Subgroup analysis (example: odd vs even studies)
    print("\n[3] Subgroup Analysis")
    subgroups = ['A' if i % 2 == 0 else 'B' for i in range(len(studies))]
    sub = analyzer.subgroup_analysis(subgroups)
    print(f"  Between-group test p-value: {sub['between_group_test']['p_value']:.3f}")
