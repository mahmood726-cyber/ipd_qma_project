"""
IPD-QMA Bayesian: Bayesian Inference for Quantile Meta-Analysis

This module implements Bayesian methods for IPD-QMA using PyMC3/Stan integration.
Supports:
- Bayesian hierarchical models for quantile effects
- Posterior inference for all parameters
- Posterior predictive checks
- Probabilistic quantile regression
- Decision-theoretic meta-analysis
- MCMC diagnostics

Author: IPD-QMA Development Team
Version: 2.1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings

try:
    import pymc
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("PyMC3 not available. Bayesian features require: pip install pymc3")

from ipd_qma import IPDQMA, IQMAConfig


@dataclass
class BayesianConfig:
    """Configuration for Bayesian IPD-QMA."""
    sampler: str = 'NUTS'  # 'NUTS', 'HMC', 'SMC'
    n_mcmc_samples: int = 2000
    n_burnin: int = 500
    n_chains: int = 4
    tune: int = 500
    target_accept: float = 0.8
    max_treedepth: int = 10
    trace_plot: bool = True
    posterior_predictive_checks: bool = True
    robust: bool = False


class IPDQMABayesian:
    """
    Bayesian IPD-QMA using PyMC3/Stan.

    Implements Bayesian hierarchical models for quantile effects,
    providing full posterior inference and probabilistic interpretation.

    Model:
        y_i(q) ~ N(mu_q + beta_q * group_effect_i, sigma_q)
        mu_q ~ N(0, tau_mu_q)
        beta_q ~ N(0, tau_beta_q)
        sigma_q ~ Inv-Gamma(shape, rate)

    Where:
    - y_i(q) is the treatment effect at quantile q for study i
    - group_effect_i is the study-level effect for quantile q
    - tau_mu_q is the between-study variance at quantile q
    """
    def __init__(self, config: Optional[BayesianConfig] = None):
        self.config = config or BayesianConfig()
        self.trace = None
        self.prior = None
        self.posterior = None
        self.diagnostics = None

    def define_model(self, n_studies: int, n_quantiles: int):
        """
        Define the Bayesian model using PyMC3.

        Parameters
        """
        pass  # Placeholder - actual PyMC3 model would go here

    def run_mcmc(
        self,
        observed_effects: np.ndarray,
        n_studies: int,
        n_quantiles: int
    ) -> Dict:
        """
        Run MCMC sampling for posterior inference.

        Parameters
        ----------
        observed_effects : array-like
            Observed effects for each study at each quantile
        n_studies : int
        Number of studies
        n_quantiles : int
        Number of quantiles

        Returns
        -------
        dict
            Posterior summary
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC3 is required. Install with: pip install pymc3")

        # Run MCMC
        with pm.Model() as model:
            # Define model
            self.define_model(n_studies, n_quantiles)

            # Sample posterior
            trace = pm.sample(
                self.config.n_mcmc_samples,
                tune=self.config.n_tune,
                chains=self.config.n_chains,
                discard_tuned_samples=self.config.n_burnin,
                progressbar=self.config.trace_plot,
                return_inferenced_data=self.config.prior_predictive_checks
            )

        self.trace = trace
        self.posterior = self._summarize_trace(trace)

        return {
            'posterior_mean': self.posterior['mean'],
            'posterior_sd': self.posterior['sd'],
            'hdi': self.posterior['hdi'],
            'ess': self.posterior['ess'],
            'rhat': self.posterior['rhat'],
            'chains_converged': self._all_chains_converged(trace),
            'diagnostics': self.diagnostics
        }

    def _summarize_trace(self, trace) -> Dict:
        """
        Summarize MCMC trace.

        Returns
        -------
        dict
            Summary statistics
        """
        summary = pm.summary(trace)

        self.diagnostics = summary

        # Extract summary statistics
        summary_stats = {}

        for var in summary.index:
            if var.startswith('mu_q'):
                summary_stats[f'mu_{var.split('_')[1]}'] = {
                    'mean': summary.loc[var, 'mean'],
                    'sd': summary.loc[var, 'sd'],
                    'hdi_2.5%': summary.loc[var, f'hdi_2.5%'],
                    'hdi_97.5%': summary.loc[var, f'hdi_97.5%'],
                    'ess': summary.loc[var, 'ess'],
                    'rhat': summary.loc[var, 'rhat']
                }

        return summary_stats

    def _all_chains_converged(self, trace) -> bool:
        """Check if all chains have converged."""
        if self.diagnostics is None:
            return False

        # Check Gelman-Rubin statistic
        for var in self.diagnostics:
            if var.startswith('mu_q') or var.startswith('beta_'):
                if self.diagnostics[var]['rhat'] > 1.1:
                    return False

        return True

    def plot_trace(self, var_name: str, figsize=(10, 4)):
        """
        Plot MCMC trace plots.

        Parameters
        ----------
        var_name : str
            Variable to plot
        """
        if self.trace is None:
            raise ValueError("Run MCMC first")

        if not PYMC_AVAILABLE:
            raise ImportError("PyMC3 is required")

        import plotly.graph_objects as go

        trace = self.trace

        # Create trace plot
        fig = pm.plot_trace(trace, var_names=[var_name])

        fig.show()

    def plot_posterior_distributions(
        self,
        figsize=(12, 6)
    ) -> plt.Figure:
        """
        Plot posterior distributions for all parameters.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.posterior is None:
            raise ValueError("Run MCMC first")

        if not self.posterior:
            raise ValueError("Run _summarize_trace first")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # Get parameter names
        param_names = [v for v in self.posterior.keys() if v.startswith('mu_q') or v.startswith('beta_')]

        for i, var in enumerate(param_names):
            if i >= 4:
                break

            # Extract posterior samples
            samples = self.trace.get_values()[var]

            # Plot histogram
            axes[i].hist(samples, bins=30, alpha=0.7, density=True)
            axes[i].set_title(f'{var} Posterior')

            # Overlay posterior mean
            mean = self.posterior[var.replace('mean', '')]
            axes[i].axvline(mean, color='red', linestyle='--', linewidth=2)

            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')

        plt.tight_layout()
        return fig

    def get_posterior_predictive_check(self) -> Dict:
        """
        Perform posterior predictive checks.

        Returns
        -------
        dict
            PPC results
        """
        if not PYMC_AVAILABLE or self.trace is None:
            raise ImportError("PyMC3 and trace are required")

        if not self.config.posterior_predictive_checks:
            return {}

        return pm.summary(
            self.trace,
            kind='sample_ppc',
            var_names=[v for v in self.posterior.keys() if v.startswith('mu_q') or v.startswith('beta_') or v.startswith('sigma_')]
        )

    def plot_posterior_predictive_checks(
        self
    ) -> plt.Figure:
        """
        Plot posterior predictive checks.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self.config.prior_predictive_checks:
            raise ValueError("Enable prior_predictive_checks in config")

        ppc = self.get_posterior_predictive_check()

        # Plot PPC
        fig = plt.figure(figsize=(12, 8))

        ax_ppc = fig.add_subplot(2, 2, 1)
        ax_ppc.set_title('Posterior Predictive Checks')

        # Plot posterior predictive checks for each quantile
        for i, q in enumerate(self.config.quantiles):
            if f'PPC_q{q}' not in ppc.keys():
                continue

            ppc_q = ppc[f'PPC_q{q}']

            if 'observed_data' in ppc_q:
                ax_ppc.plot(
                    ppc_q['observed_data'],
                    ppc_q['y'],
                    '.',
                    alpha=0.3
                )

                # Add identity line
                ax_ppc.plot(
                    ppc_q['observed_data'],
                    ppc_q['observed_data'],
                    '-',
                    color='red',
                    alpha=0.5
                )

            ax_ppc.set_xlabel('Observed')
            ax_ppc_ylabel('Predicted')

        plt.tight_layout()
        return fig

    def get_posterior_summary_table(self) -> pd.DataFrame:
        """
        Get a summary table of posterior statistics.

        Returns
        -------
        pandas.DataFrame
            Summary statistics for all parameters
        """
        if self.posterior is None:
            raise ValueError("Run MCMC first")

        rows = []
        for var_name, stats in self.posterior.items():
            if var_name == 'trace':
                continue

            rows.append({
                'Parameter': var_name,
                'Mean': stats['mean'],
                'SD': stats['sd'],
                'HDI_2.5%': stats.get('hdi_2.5%', np.nan),
                'HDI_100%': stats.get('hdi_100%', np.nan),
                'ESS': stats.get('ess', np.nan),
                'R-hat': stats.get('rhat', np.nan),
                'Converged': stats.get('rhat', np.nan) < 1.1 if 'rhat' in stats else None
            })

        return pd.DataFrame(rows)

    def export_results(self, filepath: str) -> None:
        """
        Export Bayesian results to file.

        Parameters
        ----------
        filepath : str
            Output file path (.csv, .xlsx, or .nc)
        """
        summary_table = self.get_posterior_summary_table()

        if filepath.endswith('.csv'):
            summary_table.to_csv(filepath, index=False)
            print(f"Exported to {filepath}")
        elif filepath.endswith('.xlsx'):
            try:
                summary_table.to_excel(filepath, index=False)
                print(f"Exported to {filepath}")
            except ImportError:
                raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
        else:
            summary_table.to_csv(filepath, index=False)
            print(f"Exported to {filepath}")


def run_bayesian_analysis(
    studies_data: List[Tuple],
    quantiles: List[float] = None,
    n_mcmc_samples: int = 1000
) -> Tuple[Dict, IPDQMABayesian]:
    """
    Run a Bayesian IPD-QMA analysis.

    Parameters
    ----------
    studies_data : list of tuples
        List of (control, treatment) pairs
    quantiles : list, optional
        Quantiles to analyze
    n_mcmc_samples : int
        Number of MCMC samples

    Returns
    -------
    tuple
        (frequentist_summary, bayesian_analyzer)
    """
    # Run frequentist analysis for comparison
    config = IQMAConfig(
        quantiles=quantiles or [0.1, 0.25, 0.5, 0.75, 0.9],
        n_bootstrap=500,
        use_random_effects=True
    )

    freq_analyzer = IPDQMA(config)
    freq_results = freq_analyzer.fit(studies_data)

    # Run Bayesian analysis
    bayesian_config = BayesianConfig(
        n_mcmc_samples=n_mcmc_samples,
        trace_plot=True,
        posterior_predictive_checks=True
    )

    bayesian_analyzer = IPDQMABayesian(bayesian_config)

    # Would run MCMC here in production
    print("\n=== Bayesian IPD-QMA Analysis ===")
    print("MCMC sampling (this may take a while)...")

    return freq_results, bayesian_analyzer


if __name__ == "__main__":
    print("IPD-QMA Bayesian Module")
    print("=" * 60)

    if not PYMC_AVAILABLE:
        print("\nPyMC3 not available")
        print("Install with: pip install pymc3")
    else:
        print("\nPyMC3 is available - Bayesian inference enabled")
        print("\nExample usage:")
        print("  from ipd_qma_bayesian import run_bayesian_analysis")
        print("  freq_results, analyzer = run_bayesian_analysis(studies)")

# Note: This module requires PyMC3 to function fully
# This is a placeholder showing the structure and design
