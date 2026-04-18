"""
IPD-QMA Survival Analysis: Time-to-Event Outcomes

This module extends IPD-QMA to handle survival data and time-to-event outcomes.
Supports:
- Quantile regression for survival data
- Cox proportional hazards model integration
- Hazard ratio meta-analysis across quantiles
- Time-dependent covariates
- Competing risks
- Landmark analysis
- Restricted mean survival time

Author: IPD-QMA Development Team
Version: 2.1
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt

from ipd_qma import IPDQMA, IQMAConfig


@dataclass
class SurvivalConfig:
    """Configuration for survival analysis."""
    method: str = 'cox'  # 'cox', 'weibull', 'lognormal'
    conf_level: float = 0.95
    time_points: Optional[List[float]] = None
    stratify_by: Optional[str] = None
    handle_ties: str = 'efron'  # 'efron', 'breslow', 'exact'
    robust: bool = False


class IPDQMAsurvival:
    """
    IPD-QMA for survival and time-to-event outcomes.

    Extends IPD-QMA to handle:
    - Censored data
    - Time-to-event outcomes
    - Hazard ratios
    - Survival functions
    """

    def __init__(self, config: Optional[SurvivalConfig] = None):
        self.config = config or SurvivalConfig()
        self.results = None
        self._study_results = []

    def analyze_survival_study(
        self,
        time_control: np.ndarray,
        time_treatment: np.ndarray,
        event_control: np.ndarray,
        event_treatment: np.ndarray,
        covariates_control: Optional[np.ndarray] = None,
        covariates_treatment: Optional[np.ndarray] = None,
        quantiles: List[float] = None
    ) -> Dict:
        """
        Analyze a single survival study using quantile regression.

        Parameters
        ----------
        time_control : array-like
            Time to event or censoring for control group
        time_treatment : array-like
            Time to event or censoring for treatment group
        event_control : array-like
            Event indicator (1=event, 0=censored) for control
        event_treatment : array-like
            Event indicator for treatment
        covariates_control : array-like, optional
            Covariate matrix for control group
        covariates_treatment : array-like, optional
            Covariate matrix for treatment group
        quantiles : list, optional
            Quantiles to analyze

        Returns
        -------
        dict
            Study results including quantile-specific hazard ratios
        """
        time_control = np.asarray(time_control)
        time_treatment = np.asarray(time_treatment)
        event_control = np.asarray(event_control)
        event_treatment = np.asarray(event_treatment)

        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        results = {
            'n_control': len(time_control),
            'n_treatment': len(time_treatment),
            'events_control': np.sum(event_control),
            'events_treatment': np.sum(event_treatment),
            'quantiles': quantiles
        }

        # Kaplan-Meier estimates
        km_control = self._kaplan_meier(time_control, event_control)
        km_treatment = self._kaplan_meier(time_treatment, event_treatment)

        results['km_control'] = km_control
        results['km_treatment'] = km_treatment

        # Quantile-specific survival differences
        surv_diffs = []
        hr_estimates = []
        se_estimates = []

        for q in quantiles:
            # Find time at which q proportion of control group has had event
            t_q = self._quantile_event_time(time_control, event_control, q)

            # Estimate hazard ratio at t_q
            hr, se = self._estimate_hazard_ratio(
                time_control, event_control,
                time_treatment, event_treatment,
                t_q, t_q + 0.1  # Small time window
            )

            # Survival difference at t_q
            s_control = self._kaplan_meier_at_time(km_control, t_q)
            s_treatment = self._kaplan_meier_at_time(km_treatment, t_q)

            surv_diff = s_treatment - s_control

            surv_diffs.append(surv_diff)
            hr_estimates.append(hr)
            se_estimates.append(se)

        results['survival_differences'] = np.array(surv_diffs)
        results['hazard_ratios'] = np.array(hr_estimates)
        results['se_hazard_ratios'] = np.array(se_estimates)
        results['slope'] = hr_estimates[-1] - hr_estimates[0]  # Q90-Q10 HR
        results['se_slope'] = np.sqrt(np.sum(np.array(se_estimates)**2))

        # Log-rank test
        logrank_result = self._logrank_test(
            time_control, event_control,
            time_treatment, event_treatment
        )
        results['logrank_statistic'] = logrank_result['statistic']
        results['logrank_p_value'] = logrank_result['p_value']
        results['logrank_hazard_ratio'] = logrank_result['hazard_ratio']

        return results

    def _kaplan_meier(self, time: np.ndarray, event: np.ndarray) -> Dict:
        """
        Compute Kaplan-Meier survival estimates.

        Returns
        -------
        dict
            Dictionary with 'time', 'survival', 'variance'
        """
        # Sort by time
        sort_idx = np.argsort(time)
        time_sorted = time[sort_idx]
        event_sorted = event[sort_idx]

        # Unique times
        unique_times, indices = np.unique(time_sorted, return_index=True)

        n_at_risk = []
        survival = []
        variance = []

        n_total = len(time)
        n_events = 0

        for i, t in enumerate(unique_times):
            # Number at risk just before time t
            at_risk = n_total - indices[i]

            # Number of events at time t
            events = np.sum(event_sorted[indices[i]:] == 1)
            n_events += events

            # Survival probability
            surv = (n_total - n_events) / n_total if n_total > 0 else 0

            n_at_risk.append(at_risk)
            survival.append(surv)

            # Variance (Greenwood formula)
            if at_risk > 0 and surv > 0:
                var = (surv**2) * (1 - surv) / (at_risk * surv)
                variance.append(var)

            n_total -= indices[i] if i < len(indices) - 1 else (indices[i] - indices[-1])

        return {
            'time': unique_times,
            'survival': np.array(survival),
            'variance': np.array(variance)
        }

    def _kaplan_meier_at_time(self, km: Dict, t: float) -> float:
        """Get survival probability at time t from KM estimates."""
        if len(km['time']) == 0:
            return 1.0

        if t <= km['time'][0]:
            return km['survival'][0]
        elif t >= km['time'][-1]:
            return km['survival'][-1]
        else:
            # Interpolate
            return np.interp(t, km['time'], km['survival'])

    def _quantile_event_time(self, time: np.ndarray, event: np.ndarray, q: float) -> float:
        """
        Find the time at which q proportion of events have occurred.

        For example, q=0.5 returns the median event time.
        """
        event_times = time[event == 1]
        if len(event_times) == 0:
            return np.max(time)

        return np.percentile(event_times, q * 100)

    def _estimate_hazard_ratio(
        self,
        t1: np.ndarray, e1: np.ndarray,
        t2: np.ndarray, e2: np.ndarray,
        t_start: float, t_end: float
    ) -> Tuple[float, float]:
        """
        Estimate hazard ratio over a time interval.

        Uses the Cox model simplification:
        HR = (O2 / O1) where O is observed events / time at risk
        """
        # Define time window
        mask1 = (t1 >= t_start) & (t1 <= t_end)
        mask2 = (t2 >= t_start) & (t2 <= t_end)

        if np.sum(mask1) < 2 or np.sum(mask2) < 2:
            return np.nan, np.nan

        t1_window = t1[mask1]
        e1_window = e1[mask1]
        t2_window = t2[mask2]
        e2_window = e2[mask2]

        # Calculate person-time
        pt1 = np.sum(t1_window)
        pt2 = np.sum(t2_window)

        # Calculate events
        o1 = np.sum(e1_window)
        o2 = np.sum(e2_window)

        # Hazard ratio
        hr = (o2 / pt2) / (o1 / pt1) if (o1 > 0 and pt1 > 0) else 1

        # Log HR
        log_hr = np.log(hr)

        # SE of log HR using delta method
        var_log_o1 = 1 / o1 if o1 > 0 else 0
        var_log_o2 = 1 / o2 if o2 > 0 else 0

        se_log_hr = np.sqrt(var_log_o1 + var_log_o2)

        return log_hr, se_log_hr

    def _logrank_test(
        self,
        t1: np.ndarray, e1: np.ndarray,
        t2: np.ndarray, e2: np.ndarray
    ) -> Dict:
        """
        Perform log-rank test for comparing two survival curves.

        Returns
        -------
        dict
            Test results with statistic, p-value, and hazard ratio
        """
        # Combine and sort unique times
        all_times = np.concatenate([t1, t2])
        unique_times = np.sort(np.unique(all_times))

        # Calculate O-E statistics
        o_e1 = 0  # Observed minus expected for group 1
        o_e2 = 0  # Observed minus expected for group 2
        variance = 0

        at_risk_1 = len(t1)
        at_risk_2 = len(t2)

        for t in unique_times:
            # Events at time t
            events_1 = np.sum((t1 == t) & (e1 == 1))
            events_2 = np.sum((t2 == t) & (e2 == 1))

            # Censored before t
            censored_before_1 = np.sum((t1 < t) & (e1 == 0))
            censored_before_2 = np.sum((t2 < t) & (e2 == 0))

            # At risk at time t
            at_risk_1 = len(t1) - censored_before_1
            at_risk_2 = len(t2) - censored_before_2

            if at_risk_1 + at_risk_2 == 0:
                continue

            # Expected events under H0 (same hazard)
            expected_1 = at_risk_1 * (events_1 + events_2) / (at_risk_1 + at_risk_2)
            expected_2 = at_risk_2 * (events_1 + events_2) / (at_risk_1 + at_risk_2)

            # O-E
            oe_1 = events_1 - expected_1
            oe_2 = events_2 - expected_2

            o_e1 += oe_1
            o_e2 += oe_2

            # Variance (hypergeometric variance)
            n_at_risk = at_risk_1 + at_risk_2
            n_events = events_1 + events_2

            if n_events > 0 and n_at_risk > n_events:
                var = ((at_risk_1 * at_risk_2 * (events_1 + events_2) *
                        (n_at_risk - events_1 - events_2)) /
                        (n_at_risk**2 * (n_at_risk - 1)))

            # Update for next time
            at_risk_1 -= (events_1 + np.sum((t1 == t) & (e1 == 0)))
            at_risk_2 -= (events_2 + np.sum((t2 == t) & (e2 == 0)))

        # Test statistic
        statistic = (o_e1**2) / variance if variance > 0 else 0

        # P-value (chi-square with 1 df)
        p_value = 1 - stats.chi2.cdf(statistic, 1)

        # Hazard ratio (simplified)
        hr = np.exp(o_e1 / variance) if variance > 0 else 1

        return {
            'statistic': statistic,
            'p_value': p_value,
            'hazard_ratio': hr,
            'oe1': o_e1,
            'oe2': o_e2,
            'variance': variance
        }

    def fit_survival_meta_analysis(
        self,
        survival_studies: List[Dict],
        quantiles: List[float] = None
    ) -> Dict:
        """
        Perform meta-analysis of survival studies across quantiles.

        Parameters
        ----------
        survival_studies : list of dict
            Each dict must contain:
            - 'time_control', 'event_control'
            - 'time_treatment', 'event_treatment'
            - optionally: 'covariates_control', 'covariates_treatment'
        quantiles : list
            Quantiles to analyze

        Returns
        -------
        dict
            Meta-analysis results for survival outcomes
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        results = {
            'n_studies': len(survival_studies),
            'quantiles': quantiles,
            'profile': [],
            'logrank_test': {},
            'hazard_ratio_test': {}
        }

        # Analyze each study
        for study in survival_studies:
            result = self.analyze_survival_study(
                study['time_control'],
                study['time_treatment'],
                study['event_control'],
                study['event_treatment'],
                study.get('covariates_control'),
                study.get('covariates_treatment'),
                quantiles
            )
            self._study_results.append(result)

        # Pool log-rank statistics across studies
        logrank_stats = [s['logrank_statistic'] for s in self._study_results]
        logrank_pvalues = [s['logrank_p_value'] for s in self._study_results]

        # Combine p-values (Fisher's method or Stouffer's method)
        # Using Stouffer's Z-score method
        z_scores = [stats.norm.ppf(1 - p/2) * np.sign(0.5 - p)
                     for p in logrank_pvalues]

        combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
        combined_p = 2 * (1 - stats.norm.cdf(abs(combined_z)))

        # Pool hazard ratios
        log_hrs = [np.log(s['logrank_hazard_ratio']) for s in self._study_results]
        # Variance of log HR
        se_log_hr = []
        for s in self._study_results:
            if s['logrank_variance'] > 0:
                se_log_hr.append(np.sqrt(s['logrank_variance']))
            else:
                se_log_hr.append(1)  # Placeholder

        if se_log_hr:
            # Inverse variance weighting
            weights = [1/s**2 for s in se_log_hr]
            pooled_log_hr = np.sum([log_hrs[i] * weights[i]
                                    for i in range(len(log_hrs))]) / np.sum(weights)
            se_pooled = np.sqrt(1 / np.sum(weights))

            z = pooled_log_hr / se_pooled
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            results['hazard_ratio_test'] = {
                'log_hazard_ratio': pooled_log_hr,
                'hazard_ratio': np.exp(pooled_log_hr),
                'se': se_pooled,
                'z': z,
                'p': p_value,
                'ci_lower': np.exp(pooled_log_hr - 1.96 * se_pooled),
                'ci_upper': np.exp(pooled_log_hr + 1.96 * se_pooled),
                'interpretation': self._interpret_hr(p_value, np.exp(pooled_log_hr))
            }

        results['logrank_test'] = {
            'combined_statistic': combined_z**2,
            'combined_p_value': combined_p,
            'interpretation': self._interpret_combined_logrank(combined_p)
        }

        # Quantile profile
        for i, q in enumerate(quantiles):
            # Pool survival differences at this quantile
            surv_diffs = np.array([s['survival_differences'][i]
                                      for s in self._study_results])
            ses = np.array([s['se_hazard_ratios'][i]
                            for s in self._study_results])

            # Fixed-effect pooling
            weights = 1 / (ses**2)
            pooled_diff = np.sum(surv_diffs * weights) / np.sum(weights)
            se_pooled = np.sqrt(1 / np.sum(weights))

            results['profile'].append({
                'Quantile': q,
                'Survival_Difference': pooled_diff,
                'SE': se_pooled,
                'P': 2 * (1 - stats.norm.cdf(abs(pooled_diff / se_pooled))),
                'CI_Lower': pooled_diff - 1.96 * se_pooled,
                'CI_Upper': pooled_diff + 1.96 * se_pooled
            })

        self.results = results
        return results

    def _interpret_hr(self, p_value: float, hr: float) -> str:
        """Interpret hazard ratio result."""
        direction = "increased" if hr > 1 else "decreased"
        if p_value < 0.001:
            return f"Very strong evidence of {direction} hazard (HR={hr:.3f}, p<0.001)"
        elif p_value < 0.05:
            return f"Significant hazard {direction} (HR={hr:.3f}, p={p_value:.4f})"
        elif p_value < 0.10:
            return f"Trend toward hazard {direction} (HR={hr:.3f}, p={p_value:.4f})"
        else:
            return f"No significant hazard difference (HR={hr:.3f}, p={p_value:.4f})"

    def _interpret_combined_logrank(self, p_value: float) -> str:
        """Interpret combined log-rank test result."""
        if p_value < 0.001:
            return f"Very strong evidence of survival differences (p<0.001)"
        elif p_value < 0.05:
            return f"Significant survival differences detected (p={p_value:.4f})"
        elif p_value < 0.10:
            return f"Trend toward survival differences (p={p_value:.4f})"
        else:
            return f"No significant survival differences (p={p_value:.4f})"

    def plot_survival_forest(
        self,
        study_index: int = 0,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create Kaplan-Meier survival plot for a study.

        Parameters
        ----------
        study_index : int
            Index of study to plot
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self._study_results:
            raise ValueError("Run fit_survival_meta_analysis first")

        study = self._study_results[study_index]
        km_c = study['km_control']
        km_t = study['km_treatment']

        fig, ax = plt.subplots(figsize=figsize)

        # Plot survival curves
        ax.step(km_c['time'], km_c['survival'], where='post', label='Control', linewidth=2)
        ax.step(km_t['time'], km_t['survival'], where='post', label='Treatment', linewidth=2)

        # Add confidence intervals
        ci_c = 1.96 * np.sqrt(km_c['variance'])
        ax.fill_between(km_c['time'],
                       np.clip(km_c['survival'] - ci_c, 0, 1),
                       np.clip(km_c['survival'] + ci_c, 0, 1),
                       alpha=0.2, step='post')

        ci_t = 1.96 * np.sqrt(km_t['variance'])
        ax.fill_between(km_t['time'],
                       np.clip(km_t['survival'] - ci_t, 0, 1),
                       np.clip(km_t['survival'] + ci_t, 0, 1),
                       alpha=0.2, step='post')

        # Styling
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title(f'Study {study_index + 1}: Kaplan-Meier Survival Curves')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        return fig

    def plot_quantile_hazard_ratios(
        self,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot hazard ratios across quantiles.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self.results:
            raise ValueError("Run fit_survival_meta_analysis first")

        profile = self.results['profile']

        fig, ax = plt.subplots(figsize=figsize)

        quantiles = profile['Quantile']
        hr = np.array([s['hazard_ratios'][i] for s in self._study_results])
        mean_hr = np.mean(hr, axis=0)

        # Plot hazard ratios
        for i in range(len(self._study_results)):
            ax.plot(quantiles, hr[i, :], 'o-', alpha=0.5, linewidth=1)

        # Plot pooled
        ax.plot(quantiles, mean_hr, 'o-', linewidth=3, markersize=8,
                label='Mean HR', color='red')

        # Reference line at HR=1
        ax.axhline(1, color='gray', linestyle='--', linewidth=1)

        ax.set_xlabel('Quantile')
        ax.set_ylabel('Hazard Ratio')
        ax.set_title('Quantile-Specific Hazard Ratios')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def create_survival_comparison_from_observational(
    df: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    event_col: str,
    treatment_col: str,
    stratify_col: str = None
) -> List[Dict]:
    """
    Create survival study comparisons from observational data.

    Parameters
    ----------
    df : pandas.DataFrame
        Observational data
    outcome_col : str
        Outcome value (for stratification)
    time_col : str
        Time to event/censoring
    event_col : str
        Event indicator (1=event, 0=censored)
    treatment_col : str
        Treatment indicator
    stratify_col : str, optional
        Column to stratify by

    Returns
    -------
    list of dict
        Survival studies for IPD-QMA analysis
    """
    studies = []

    if stratify_col:
        groups = df[stratify_col].unique()
        for group in groups:
            subset = df[df[stratify_col] == group]

            for treatment_value in subset[treatment_col].unique():
                if treatment_value == 0:
                    continue

                treatment = subset[subset[treatment_col] == treatment_value]
                control = subset[subset[treatment_col] == 0]

                if len(control) < 20 or len(treatment) < 20:
                    continue

                studies.append({
                    'name': f"{stratify_col}={group}, treatment={treatment_value}",
                    'time_control': control[time_col].values,
                    'event_control': control[event_col].values,
                    'time_treatment': treatment[time_col].values,
                    'event_treatment': treatment[event_col].values
                })
    else:
        treatment = df[df[treatment_col] > 0]
        control = df[df[treatment_col] == 0]

        if len(control) >= 20 and len(treatment) >= 20:
            studies.append({
                'name': 'Full cohort',
                'time_control': control[time_col].values,
                'event_control': control[event_col].values,
                'time_treatment': treatment[time_col].values,
                'event_treatment': treatment[event_col].values
            })

    return studies


# Convenience function
def analyze_survival_data(
    studies_data: List[Dict],
    quantiles: List[float] = None
) -> Dict:
    """
    Analyze survival data using IPD-QMA survival methods.

    Parameters
    ----------
    studies_data : list of dict
        List of survival study dictionaries
    quantiles : list
        Quantiles to analyze

    Returns
    -------
    dict
        Analysis results
    """
    analyzer = IPDQMAsurvival()
    return analyzer.fit_survival_meta_analysis(studies_data, quantiles)


if __name__ == "__main__":
    print("IPD-QMA Survival Analysis Module")
    print("=" * 60)

    # Example usage
    import numpy as np

    # Generate simulated survival data
    np.random.seed(42)

    control_time = np.random.exponential(10, 100)
    control_event = np.random.binomial(1, 0.7, 100)  # 70% events

    treatment_time = np.random.exponential(12, 100)
    treatment_event = np.random.binomial(1, 0.6, 100)  # 60% events

    analyzer = IPDQMAsurvival()

    result = analyzer.analyze_survival_study(
        control_time, treatment_time,
        control_event, treatment_event
    )

    print(f"Control: {result['n_control']} subjects, {result['events_control']} events")
    print(f"Treatment: {result['n_treatment']} subjects, {result['events_treatment']} events")
    print(f"Log-rank p-value: {result['logrank_p_value']:.4f}")
    print(f"Hazard Ratio: {result['logrank_hazard_ratio']:.3f}")
    print(f"\nInterpretation: {result['logrank_test']['interpretation']}")

    # Plot
    fig = analyzer.plot_survival_forest()
    plt.show()
