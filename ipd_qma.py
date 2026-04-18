"""
IPD-QMA: Individual Participant Data Quantile Meta-Analysis (Improved Version)
A tool for detecting heterogeneous treatment effects (location-scale shifts).

Author: Improved version with enhanced features
Version: 2.0 - Added parallel processing and progress tracking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
import warnings
import multiprocessing as mp
from functools import partial
import os

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def _bootstrap_worker(args: Tuple[np.ndarray, np.ndarray, List[float], int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Worker function for parallel bootstrap computation.

    Parameters
    ----------
    args : tuple
        (control_data, treatment_data, quantiles, n_samples)

    Returns
    -------
    tuple
        (boot_quantiles_control, boot_quantiles_treatment)
    """
    control, treatment, quantiles, n_samples = args
    n_c, n_t = len(control), len(treatment)

    # Generate bootstrap indices
    idx_c = np.random.randint(0, n_c, (n_samples, n_c))
    idx_t = np.random.randint(0, n_t, (n_samples, n_t))

    boot_c = control[idx_c]
    boot_t = treatment[idx_t]

    # Calculate quantiles
    boot_q_c = np.percentile(boot_c, [q * 100 for q in quantiles], axis=1)
    boot_q_t = np.percentile(boot_t, [q * 100 for q in quantiles], axis=1)

    return boot_q_c, boot_q_t


@dataclass
class IQMAConfig:
    """Configuration for IPD-QMA analysis."""
    quantiles: List[float] = None
    n_bootstrap: int = 200
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    use_random_effects: bool = True
    tau2_estimator: str = 'dl'  # 'dl' (DerSimonian-Laird), 'reml', 'pm'
    n_workers: Optional[int] = None  # Number of parallel workers (None = cpu_count)
    show_progress: bool = True  # Show progress bars
    parallel_threshold: int = 1000  # Minimum bootstrap samples for parallel processing

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]


class IPDQMA:
    """
    IPD-QMA: Individual Participant Data Quantile Meta-Analysis.

    Detects heterogeneous treatment effects across patient severity
    distributions using quantile-based analysis with bootstrap inference.

    Parameters
    ----------
    config : IQMAConfig
        Configuration object for the analysis
    quantiles : list of float, optional
        Quantiles to analyze (default: [0.1, 0.25, 0.5, 0.75, 0.9])
    n_boot : int, optional
        Number of bootstrap samples (default: 200)

    Examples
    --------
    >>> analyzer = IPDQMA(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], n_boot=500)
    >>> studies = [(control1, treatment1), (control2, treatment2)]
    >>> results = analyzer.fit(studies)
    >>> analyzer.plot()
    """

    def __init__(
        self,
        config: Optional[IQMAConfig] = None,
        quantiles: Optional[List[float]] = None,
        n_boot: int = 200
    ):
        self.config = config or IQMAConfig()
        if quantiles is not None:
            self.config.quantiles = quantiles
        # Only override n_bootstrap if config wasn't provided
        if config is None:
            self.config.n_bootstrap = n_boot

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        self.quantiles = self.config.quantiles
        self.n_boot = self.config.n_bootstrap
        self.results = None
        self._study_results = []

    def _validate_inputs(self, control: np.ndarray, treatment: np.ndarray) -> None:
        """Validate input data for a single study."""
        if len(control) < 10:
            warnings.warn(f"Small control sample size (n={len(control)}). Bootstrap may be unstable.")
        if len(treatment) < 10:
            warnings.warn(f"Small treatment sample size (n={len(treatment)}). Bootstrap may be unstable.")
        if np.any(np.isnan(control)) or np.any(np.isnan(treatment)):
            raise ValueError("Data contains NaN values.")
        if np.any(np.isinf(control)) or np.any(np.isinf(treatment)):
            raise ValueError("Data contains infinite values.")

    def analyze_study(
        self,
        control: Union[np.ndarray, List],
        treatment: Union[np.ndarray, List],
        show_progress: Optional[bool] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze a single study using bootstrap quantile estimation.

        Parameters
        ----------
        control : array-like
            Control group outcomes
        treatment : array-like
            Treatment group outcomes
        show_progress : bool, optional
            Override progress bar setting from config

        Returns
        -------
        dict
            Dictionary containing effect estimates and standard errors
        """
        control = np.asarray(control)
        treatment = np.asarray(treatment)

        self._validate_inputs(control, treatment)

        n_c, n_t = len(control), len(treatment)

        # Determine whether to use parallel processing
        use_parallel = (
            self.n_boot >= self.config.parallel_threshold and
            self.config.n_workers != 1 and
            mp.cpu_count() > 1
        )

        # Determine progress bar setting
        should_show_progress = show_progress if show_progress is not None else self.config.show_progress
        should_show_progress = should_show_progress and TQDM_AVAILABLE

        # Bootstrap computation
        if use_parallel:
            boot_q_c, boot_q_t = self._parallel_bootstrap(control, treatment, should_show_progress)
        else:
            boot_q_c, boot_q_t = self._vectorized_bootstrap(control, treatment, should_show_progress)

        # Differences (Effects) - shape: (n_quantiles, n_boot)
        boot_diffs = boot_q_t - boot_q_c

        # Point estimates (observed quantiles)
        obs_q_c = np.percentile(control, [q * 100 for q in self.quantiles])
        obs_q_t = np.percentile(treatment, [q * 100 for q in self.quantiles])
        obs_effects = obs_q_t - obs_q_c

        # Standard errors from bootstrap
        se_effects = np.std(boot_diffs, axis=1, ddof=1)

        # Slope (Q90 - Q10): measures effect heterogeneity across quantiles
        boot_slopes = boot_diffs[-1, :] - boot_diffs[0, :]
        obs_slope = obs_effects[-1] - obs_effects[0]
        se_slope = np.std(boot_slopes, ddof=1)

        # Log Variance Ratio (lnVR): measures scale shift
        var_t = np.var(treatment, ddof=1)
        var_c = np.var(control, ddof=1)
        lnvr = np.log(var_t / var_c) if var_c > 0 and var_t > 0 else 0
        se_lnvr = np.sqrt(1 / (n_t - 1) + 1 / (n_c - 1))

        # Add bias-corrected estimates (BCa method approximation)
        bias_correction = self._calculate_bias_correction(boot_diffs, obs_effects)

        return {
            'quantiles': obs_effects,
            'se_quantiles': se_effects,
            'quantile_effects_bc': obs_effects - bias_correction,
            'slope': obs_slope,
            'se_slope': se_slope,
            'lnvr': lnvr,
            'se_lnvr': se_lnvr,
            'n_control': n_c,
            'n_treatment': n_t,
            'mean_control': np.mean(control),
            'mean_treatment': np.mean(treatment),
            'sd_control': np.std(control, ddof=1),
            'sd_treatment': np.std(treatment, ddof=1)
        }

    def _parallel_bootstrap(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        show_progress: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform bootstrap computation using parallel processing.

        Parameters
        ----------
        control : np.ndarray
            Control group data
        treatment : np.ndarray
            Treatment group data
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        tuple
            (boot_q_c, boot_q_t) bootstrap quantile arrays
        """
        n_workers = self.config.n_workers or mp.cpu_count()
        chunk_size = max(1, self.n_boot // n_workers)

        # Split work into chunks
        chunks = []
        remaining = self.n_boot
        for i in range(n_workers):
            size = min(chunk_size, remaining)
            if size == 0:
                break
            chunks.append((control, treatment, self.quantiles, size))
            remaining -= size

        # Use multiprocessing with tqdm progress bar
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(_bootstrap_worker, chunks),
                    total=len(chunks),
                    desc="Bootstrap",
                    unit="chunk",
                    disable=not show_progress
                ))
            else:
                results = pool.map(_bootstrap_worker, chunks)

        # Concatenate results
        boot_q_c = np.concatenate([r[0] for r in results], axis=1)
        boot_q_t = np.concatenate([r[1] for r in results], axis=1)

        return boot_q_c, boot_q_t

    def _vectorized_bootstrap(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        show_progress: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform bootstrap computation using vectorized operations.

        Parameters
        ----------
        control : np.ndarray
            Control group data
        treatment : np.ndarray
            Treatment group data
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        tuple
            (boot_q_c, boot_q_t) bootstrap quantile arrays
        """
        n_c, n_t = len(control), len(treatment)

        # Generate all bootstrap indices at once (vectorized)
        idx_c = np.random.randint(0, n_c, (self.n_boot, n_c))
        idx_t = np.random.randint(0, n_t, (self.n_boot, n_t))

        boot_c = control[idx_c]
        boot_t = treatment[idx_t]

        # Calculate quantiles for all bootstraps
        if show_progress and TQDM_AVAILABLE:
            # Calculate with progress bar
            boot_q_c = np.zeros((len(self.quantiles), self.n_boot))
            boot_q_t = np.zeros((len(self.quantiles), self.n_boot))

            for i, q in enumerate(tqdm(self.quantiles, desc="Quantiles", disable=not show_progress)):
                boot_q_c[i, :] = np.percentile(boot_c, q * 100, axis=1)
                boot_q_t[i, :] = np.percentile(boot_t, q * 100, axis=1)
        else:
            # Vectorized calculation
            boot_q_c = np.percentile(boot_c, [q * 100 for q in self.quantiles], axis=1)
            boot_q_t = np.percentile(boot_t, [q * 100 for q in self.quantiles], axis=1)

        return boot_q_c, boot_q_t

    def _calculate_bias_correction(
        self,
        boot_diffs: np.ndarray,
        obs_effects: np.ndarray
    ) -> np.ndarray:
        """Calculate bias correction for bootstrap estimates."""
        # Proportion of bootstrap estimates less than observed
        prop_less = np.mean(boot_diffs < obs_effects[:, np.newaxis], axis=1)
        # Avoid log(0) issues
        prop_less = np.clip(prop_less, 0.001, 0.999)
        z0 = stats.norm.ppf(prop_less)
        # Bias correction approximation
        bias = -z0 * np.std(boot_diffs, axis=1, ddof=1)
        return bias

    def _pool_fixed_effect(
        self,
        estimates: np.ndarray,
        se: np.ndarray
    ) -> Dict[str, float]:
        """Fixed-effect meta-analysis using inverse variance weighting."""
        weights = 1.0 / (se ** 2)
        pooled = np.sum(estimates * weights) / np.sum(weights)
        se_pooled = np.sqrt(1.0 / np.sum(weights))
        z = pooled / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        ci_margin = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2) * se_pooled

        return {
            'estimate': pooled,
            'se': se_pooled,
            'z': z,
            'p': p_value,
            'lower': pooled - ci_margin,
            'upper': pooled + ci_margin,
            'weights': weights / np.sum(weights)
        }

    def _estimate_heterogeneity(
        self,
        estimates: np.ndarray,
        se: np.ndarray,
        method: str = 'dl'
    ) -> Dict[str, float]:
        """
        Estimate between-study heterogeneity (τ²).

        Methods
        -------
        'dl' : DerSimonian-Laird
        'pm' : Paule-Mandel
        """
        k = len(estimates)
        if k < 2:
            return {'tau2': 0, 'tau': 0, 'i2': 0, 'q': 0, 'q_p': 1}

        # Q statistic
        weights = 1.0 / (se ** 2)
        pooled_fixed = np.sum(estimates * weights) / np.sum(weights)
        q = np.sum(weights * (estimates - pooled_fixed) ** 2)
        q_p = 1 - stats.chi2.cdf(q, k - 1)

        if method == 'dl':
            # DerSimonian-Laird estimator
            c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
            tau2 = max(0, (q - (k - 1)) / c)
        elif method == 'pm':
            # Paule-Mandel estimator (iterative)
            tau2 = 0
            for _ in range(50):
                weights_pm = 1.0 / (se ** 2 + tau2)
                pooled_pm = np.sum(estimates * weights_pm) / np.sum(weights_pm)
                q_new = np.sum(weights_pm * (estimates - pooled_pm) ** 2)
                if abs(q_new - (k - 1)) < 1e-6:
                    break
                tau2 = max(0, tau2 + (q_new - (k - 1)) / (2 * np.sum(weights_pm ** 2)))
        else:
            tau2 = 0

        tau = np.sqrt(tau2)

        # I² statistic
        i2 = max(0, (q - (k - 1)) / q * 100) if q > (k - 1) else 0

        return {
            'tau2': tau2,
            'tau': tau,
            'i2': i2,
            'q': q,
            'q_p': q_p,
            'k': k
        }

    def _pool_random_effects(
        self,
        estimates: np.ndarray,
        se: np.ndarray,
        het: Dict[str, float]
    ) -> Dict[str, float]:
        """Random-effects meta-analysis using DL estimator."""
        tau2 = het['tau2']

        # Re-weight with between-study variance
        weights_re = 1.0 / (se ** 2 + tau2)
        pooled = np.sum(estimates * weights_re) / np.sum(weights_re)
        se_pooled = np.sqrt(1.0 / np.sum(weights_re))

        z = pooled / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        ci_margin = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2) * se_pooled

        # Prediction interval
        se_pred = np.sqrt(se_pooled ** 2 + tau2)
        pred_margin = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2) * se_pred

        return {
            'estimate': pooled,
            'se': se_pooled,
            'z': z,
            'p': p_value,
            'lower': pooled - ci_margin,
            'upper': pooled + ci_margin,
            'pred_lower': pooled - pred_margin,
            'pred_upper': pooled + pred_margin,
            'weights': weights_re / np.sum(weights_re)
        }

    def fit(self, studies_data: List[Tuple[Union[np.ndarray, List], Union[np.ndarray, List]]]) -> Dict:
        """
        Fit the IPD-QMA model to multiple studies.

        Parameters
        ----------
        studies_data : list of tuples
            Each tuple contains (control_outcomes, treatment_outcomes) for a study

        Returns
        -------
        dict
            Results dictionary containing quantile profiles, heterogeneity tests,
            and overall effect estimates
        """
        # Analyze each study with progress bar
        if self.config.show_progress and TQDM_AVAILABLE and len(studies_data) > 1:
            self._study_results = [
                self.analyze_study(c, t, show_progress=False)
                for c, t in tqdm(studies_data, desc="Analyzing studies", unit="study")
            ]
        else:
            self._study_results = [self.analyze_study(c, t) for c, t in studies_data]

        n_studies = len(self._study_results)

        # Pool quantiles with progress bar
        q_summary = []
        quantile_iter = tqdm(self.quantiles, desc="Pooling quantiles", disable=not (self.config.show_progress and TQDM_AVAILABLE)) if self.config.show_progress and TQDM_AVAILABLE else self.quantiles
        for i, q in enumerate(quantile_iter):
            est = np.array([s['quantiles'][i] for s in self._study_results])
            se = np.array([s['se_quantiles'][i] for s in self._study_results])

            if self.config.use_random_effects:
                het = self._estimate_heterogeneity(est, se, self.config.tau2_estimator)
                result = self._pool_random_effects(est, se, het)
            else:
                result = self._pool_fixed_effect(est, se)
                het = self._estimate_heterogeneity(est, se)

            q_summary.append({
                'Quantile': q,
                'Effect': result['estimate'],
                'SE': result['se'],
                'Z': result.get('z', 0),
                'P': result['p'],
                'CI_Lower': result['lower'],
                'CI_Upper': result['upper'],
                'Pred_Lower': result.get('pred_lower', np.nan),
                'Pred_Upper': result.get('pred_upper', np.nan),
                'I2': het.get('i2', 0),
                'Tau2': het.get('tau2', 0),
                'Q': het.get('q', 0),
                'Q_P': het.get('q_p', 1)
            })

        # Pool slope
        s_est = np.array([s['slope'] for s in self._study_results])
        s_se = np.array([s['se_slope'] for s in self._study_results])
        s_het = self._estimate_heterogeneity(s_est, s_se)

        if self.config.use_random_effects:
            slope_result = self._pool_random_effects(s_est, s_se, s_het)
        else:
            slope_result = self._pool_fixed_effect(s_est, s_se)

        # Pool lnVR
        l_est = np.array([s['lnvr'] for s in self._study_results])
        l_se = np.array([s['se_lnvr'] for s in self._study_results])
        l_het = self._estimate_heterogeneity(l_est, l_se)

        if self.config.use_random_effects:
            lnvr_result = self._pool_random_effects(l_est, l_se, l_het)
        else:
            lnvr_result = self._pool_fixed_effect(l_est, l_se)

        # Create results dictionary
        self.results = {
            'n_studies': n_studies,
            'model_type': 'random_effects' if self.config.use_random_effects else 'fixed_effect',
            'profile': pd.DataFrame(q_summary),
            'slope_test': {
                'estimate': slope_result['estimate'],
                'se': slope_result['se'],
                'p': slope_result['p'],
                'ci_lower': slope_result['lower'],
                'ci_upper': slope_result['upper'],
                'i2': s_het['i2'],
                'tau2': s_het['tau2'],
                'q_p': s_het['q_p'],
                'interpretation': self._interpret_slope(slope_result['estimate'], slope_result['p'])
            },
            'lnvr_test': {
                'estimate': lnvr_result['estimate'],
                'se': lnvr_result['se'],
                'p': lnvr_result['p'],
                'ci_lower': lnvr_result['lower'],
                'ci_upper': lnvr_result['upper'],
                'i2': l_het['i2'],
                'tau2': l_het['tau2'],
                'q_p': l_het['q_p'],
                'interpretation': self._interpret_lnvr(lnvr_result['estimate'], lnvr_result['p'])
            },
            'study_details': self._study_results,
            'config': self.config
        }

        return self.results

    def _interpret_slope(self, estimate: float, p_value: float) -> str:
        """Interpret the slope test result."""
        if p_value < 0.001:
            return f"Very strong evidence of heterogeneous effects (slope={estimate:.3f}, p<0.001)"
        elif p_value < 0.05:
            return f"Significant heterogeneous effects detected (slope={estimate:.3f}, p={p_value:.4f})"
        elif p_value < 0.10:
            return f"Trend toward heterogeneous effects (slope={estimate:.3f}, p={p_value:.4f})"
        else:
            return f"No significant heterogeneity detected (slope={estimate:.3f}, p={p_value:.4f})"

    def _interpret_lnvr(self, estimate: float, p_value: float) -> str:
        """Interpret the lnVR test result."""
        direction = "increased" if estimate > 0 else "decreased"
        if p_value < 0.001:
            return f"Very strong evidence of variance difference (lnVR={estimate:.3f}, p<0.001)"
        elif p_value < 0.05:
            return f"Significant variance {direction} (lnVR={estimate:.3f}, p={p_value:.4f})"
        elif p_value < 0.10:
            return f"Trend toward variance difference (lnVR={estimate:.3f}, p={p_value:.4f})"
        else:
            return f"No significant variance difference (lnVR={estimate:.3f}, p={p_value:.4f})"

    def plot(self, figsize: Tuple[int, int] = (12, 6), show_predictions: bool = True) -> plt.Figure:
        """
        Generate the fan plot showing treatment effects across quantiles.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        show_predictions : bool
            Whether to show prediction intervals

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if self.results is None:
            raise ValueError("Run fit() first before plotting.")

        df = self.results['profile']

        fig, ax = plt.subplots(figsize=figsize)

        # Reference line at zero
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Plot confidence interval
        ax.fill_between(
            df['Quantile'],
            df['CI_Lower'],
            df['CI_Upper'],
            color='#2E86C1',
            alpha=0.25,
            label='95% Confidence Interval'
        )

        # Plot prediction interval if available and requested
        if show_predictions and not df['Pred_Lower'].isna().all():
            ax.fill_between(
                df['Quantile'],
                df['Pred_Lower'],
                df['Pred_Upper'],
                color='#2E86C1',
                alpha=0.15,
                label='95% Prediction Interval'
            )

        # Plot effect sizes
        ax.plot(
            df['Quantile'],
            df['Effect'],
            'o-',
            color='#2E86C1',
            linewidth=2.5,
            markersize=8,
            label='Pooled Effect'
        )

        # Add individual study points if available
        if len(self._study_results) > 1:
            for i, study in enumerate(self._study_results):
                ax.plot(
                    self.quantiles,
                    study['quantiles'],
                    '.',
                    color='gray',
                    alpha=0.3,
                    markersize=4
                )

        # Styling
        ax.set_title(
            f"IPD-QMA Profile: Treatment Effects Across Patient Severity Quantiles\n"
            f"({self.results['n_studies']} studies, {self.results['model_type'].replace('_', ' ').title()})",
            fontsize=13,
            fontweight='bold'
        )
        ax.set_xlabel("Patient Severity Quantile", fontsize=12)
        ax.set_ylabel("Treatment Effect Size", fontsize=12)
        ax.set_xticks(self.quantiles)
        ax.set_xticklabels([f'{q:.0%}' for q in self.quantiles])
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout()
        return fig

    def plot_forest(self, quantile_index: int = -1, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a forest plot for a specific quantile (default: median).

        Parameters
        ----------
        quantile_index : int
            Index of quantile to plot (default: -1 for median)
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if self.results is None:
            raise ValueError("Run fit() first before plotting.")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2
        q = self.quantiles[q_idx]

        fig, ax = plt.subplots(figsize=figsize)

        # Extract study-level data for this quantile
        study_effects = [s['quantiles'][q_idx] for s in self._study_results]
        study_ses = [s['se_quantiles'][q_idx] for s in self._study_results]
        study_names = [f"Study {i+1}" for i in range(len(self._study_results))]

        # Calculate CIs
        cis = [(e - 1.96 * se, e + 1.96 * se) for e, se in zip(study_effects, study_ses)]

        # Plot individual studies
        y_pos = np.arange(len(study_effects))
        ax.errorbar(
            study_effects,
            y_pos,
            xerr=[1.96 * np.array(study_ses), 1.96 * np.array(study_ses)],
            fmt='o',
            capsize=5,
            color='#2E86C1',
            label='Individual Studies'
        )

        # Plot pooled effect
        pooled = self.results['profile'].iloc[q_idx]
        ax.axvline(
            pooled['Effect'],
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Pooled ({self.results["model_type"].replace("_", " ").title()})'
        )

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(study_names)
        ax.set_xlabel(f"Effect Size at {q:.0%} Quantile", fontsize=12)
        ax.set_title(f"Forest Plot: {q:.0%} Quantile Treatment Effect", fontsize=13, fontweight='bold')
        ax.axvline(0, color='gray', linestyle='-', linewidth=1)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def summary(self) -> pd.DataFrame:
        """
        Generate a summary table of results.

        Returns
        -------
        pandas.DataFrame
            Summary statistics for each quantile
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        summary_data = {
            'Quantile': self.quantiles,
            'Effect': self.results['profile']['Effect'],
            'SE': self.results['profile']['SE'],
            '95% CI Lower': self.results['profile']['CI_Lower'],
            '95% CI Upper': self.results['profile']['CI_Upper'],
            'P-value': self.results['profile']['P'],
            'I² (%)': self.results['profile']['I2'].round(1),
            'τ²': self.results['profile']['Tau2'].round(4)
        }

        df = pd.DataFrame(summary_data)

        print(f"\n{'='*60}")
        print(f"IPD-QMA Analysis Summary ({self.results['n_studies']} studies)")
        print(f"{'='*60}")
        print(f"\nSlope Test (Heterogeneity):")
        print(f"  Estimate: {self.results['slope_test']['estimate']:.4f}")
        print(f"  P-value: {self.results['slope_test']['p']:.4f}")
        print(f"  I²: {self.results['slope_test']['i2']:.1f}%")
        print(f"  {self.results['slope_test']['interpretation']}")

        print(f"\nLog Variance Ratio Test (Scale Shift):")
        print(f"  Estimate: {self.results['lnvr_test']['estimate']:.4f}")
        print(f"  P-value: {self.results['lnvr_test']['p']:.4f}")
        print(f"  I²: {self.results['lnvr_test']['i2']:.1f}%")
        print(f"  {self.results['lnvr_test']['interpretation']}")
        print(f"{'='*60}\n")

        return df

    def export_results(self, filepath: str, format: str = 'xlsx') -> None:
        """
        Export results to file.

        Parameters
        ----------
        filepath : str
            Output file path
        format : str
            Output format ('xlsx' or 'csv')
        """
        if self.results is None:
            raise ValueError("Run fit() first.")

        # Summary sheet
        summary_df = self.summary()

        if format == 'xlsx':
            try:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    self.results['profile'].to_excel(writer, sheet_name='Quantile_Profile', index=False)

                    # Study details
                    study_data = []
                    for i, s in enumerate(self._study_results):
                        study_data.append({
                            'Study': i + 1,
                            'N_Control': s['n_control'],
                            'N_Treatment': s['n_treatment'],
                            'Mean_Control': s['mean_control'],
                            'Mean_Treatment': s['mean_treatment'],
                            'SD_Control': s['sd_control'],
                            'SD_Treatment': s['sd_treatment'],
                            'Slope': s['slope'],
                            'LnVR': s['lnvr']
                        })
                    pd.DataFrame(study_data).to_excel(writer, sheet_name='Study_Details', index=False)
                print(f"Results exported to {filepath}")
            except ImportError:
                raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
        else:
            summary_df.to_csv(filepath, index=False)
            print(f"Results exported to {filepath}")


def run_tutorial():
    """Run a tutorial demonstration of IPD-QMA."""
    print("=" * 70)
    print("IPD-QMA Tutorial: Heterogeneous Treatment Effect Detection")
    print("=" * 70)

    # 1. Generate simulated data with heterogeneous effects
    print("\n[1] Generating simulated data from 20 studies...")
    print("    - Simulating skewed distributions (exponential)")
    print("    - Treatment increases both mean AND variance (location + scale shift)")

    np.random.seed(42)
    data = []
    for i in range(20):
        # Base exponential distribution
        base_scale = np.random.uniform(0.8, 1.2)

        # Control group
        control = np.random.exponential(base_scale, 150) - 1

        # Treatment group: larger variance + location shift
        # This creates quantile-dependent effects
        variance_multiplier = np.random.uniform(2.5, 3.5)
        treatment = (np.random.exponential(base_scale, 150) - 1) * variance_multiplier

        data.append((control, treatment))

    print(f"    ✓ Generated {len(data)} studies")
    print(f"    - Average control sample size: {np.mean([len(c) for c, _ in data]):.0f}")
    print(f"    - Average treatment sample size: {np.mean([len(t) for _, t in data]):.0f}")

    # 2. Run analysis
    print("\n[2] Running IPD-QMA analysis...")

    config = IQMAConfig(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        n_bootstrap=500,
        use_random_effects=True,
        tau2_estimator='dl',
        random_seed=42
    )

    analyzer = IPDQMA(config)
    results = analyzer.fit(data)

    print(f"    ✓ Analysis complete")
    print(f"    - Model: {results['model_type'].replace('_', ' ').title()}")
    print(f"    - Bootstrap samples: {config.n_bootstrap}")

    # 3. Display results
    print("\n[3] RESULTS:")
    print("   " + "=" * 66)

    # Get summary table
    summary = analyzer.summary()

    # 4. Create plots
    print("\n[4] Generating visualization plots...")

    # Fan plot
    fig1 = analyzer.plot(figsize=(12, 6))
    plt.savefig('ipd_qma_fan_plot.png', dpi=150, bbox_inches='tight')
    print("    ✓ Fan plot saved to 'ipd_qma_fan_plot.png'")

    # Forest plot
    fig2 = analyzer.plot_forest(figsize=(10, 8))
    plt.savefig('ipd_qma_forest_plot.png', dpi=150, bbox_inches='tight')
    print("    ✓ Forest plot saved to 'ipd_qma_forest_plot.png'")

    # 5. Export results
    print("\n[5] Exporting results...")
    try:
        analyzer.export_results('ipd_qma_results.xlsx', format='xlsx')
    except Exception as e:
        print(f"    Note: Excel export skipped ({e})")

    print("\n" + "=" * 70)
    print("Tutorial complete! Check the generated plots and results.")
    print("=" * 70)

    plt.show()

    return analyzer, results


if __name__ == "__main__":
    run_tutorial()
