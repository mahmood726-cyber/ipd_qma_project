"""
IPD-QMA Network Meta-Analysis Module

This module implements network meta-analysis for quantile effects,
allowing simultaneous comparison of multiple treatments.

Features:
- Multi-arm trial support
- Rank preserving models
- Node-splitting models
- League tables
- Transitivity checks
- Inconsistency detection
- Network visualization

Author: IPDQMA Development Team
Version: 2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings

from ipd_qma import IPDQMA, IQMAConfig


@dataclass
class NetworkConfig:
    """Configuration for network meta-analysis."""
    network_type: str = 'network'  # 'network', 'star', 'hierarchy'
    consistency_model: str = 'rpm'  # 'rpm' (rank preserving), 'npm' (network meta-analysis), 'consistency'
    tau2_estimator: str = 'dl'
    use_intransitivity_check: bool = True
    alpha_intransitivity: float = 0.05
    use_node_splitting: bool = True
    split_alpha: float = 0.05


class IPDQMANetwork:
    """
    Network meta-analysis for IPD-QMA with quantile effects.

    Handles:
    - Multi-arm trials (2+ arms)
    - Network structure
    - Inconsistency detection
    - Transitivity assessment
    - League tables
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self.network = None
        self.results = None
        self._network_summary = {}

    def build_network(
        self,
        studies_data: List[Dict],
        study_name_col: str = 'study',
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome',
        quantiles: List[float] = None
    ) -> Dict:
        """
        Build network structure from multi-arm studies.

        Parameters
        ----------
        studies_data : list of dict
            Each dict should contain:
            - treatment_col: list of treatment names
            - outcome_col: list of outcomes for each treatment
            - Optional: sample sizes for each treatment

        Returns
        -------
        dict
            Network structure with nodes, edges, and summary
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.1, 0.75, 0.9]

        # Build adjacency list
        adjacency = defaultdict(list)
        treatments = set()
        edges = []

        for study_idx, study in enumerate(studies_data):
            treatments = study[treatment_col]
            outcomes = study[outcome_col]

            # Get number of arms
            n_arms = len(treatments)

            # Create edges for all pairwise comparisons
            for i in range(n_arms):
                for j in range(i+1, n_arms):
                    treatments.add(treatments[i])
                    treatments.add(treatments[j])

                    adjacency[treatments[i]].append(treatments[j])
                    adjacency[treatments[j]].append(treatments[i])

                    edges.append({
                        'study_id': study[study_name_col] if study_name_col in study else f"S{study_idx}",
                        'treatment1': treatments[i],
                        'treatment2': treatments[j],
                        'n_treatment1': len([outcomes[i]]),
                        'n_treatment2': len([outcomes[j]]),
                        'mean_effect': np.mean([outcomes[i], outcomes[j]]),
                        'raw_data': (outcomes[i], outcomes[j])
                    })

        return {
            'treatments': sorted(list(treatments)),
            'adjacency': dict(adjacency),
            'edges': edges,
            'n_treatments': len(treatments),
            'n_edges': len(edges)
        }

    def analyze_network(
        self,
        network: Dict,
        quantiles: List[float] = None
    ) -> Dict:
        """
        Analyze network using IPD-QMA principles.

        Parameters
        ----------
        network : dict
            Network structure from build_network()
        quantiles : list
            Quantiles to analyze

        Returns
        -------
        results : dict
            Network analysis results
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 2.0]  # Include median and log-scale

        results = {
            'n_nodes': network['n_treatments'],
            'n_edges': network['n_edges'],
            'quantiles': quantiles,
            'network_type': self.config.network_type,
            'pairwise_results': []
        }

        # Analyze each edge
        for edge in network['edges']:
            if self.config.use_node_splitting:
                # Node-splitting model
                edge_result = self._node_split_analysis(
                    edge['treatment1'], edge['treatment2'],
                    edge['raw_data'],
                    edge['n_treatment1'], edge['n_treatment2']
                )
            else:
                # Direct pooling
                edge_result = self._analyze_pairwise_direct(edge, quantiles)

            results['pairwise_results'].append({
                'comparison': f"{edge['treatment1']} vs {edge['treatment']}",
                'study': edge['study_id'],
                'n1': edge['n_treatment1'],
                'n2': edge['n_treatment2'],
                'results': edge_result
            })

        # Perform transitivity check
        if self.config.use_intransitivity_check:
            results['transitivity'] = self._check_transitivity(network, results)

        # Generate league table
        results['league_table'] = self._generate_league_table(network, results)

        # Detect inconsistencies
        if self.config.consistency_model == 'rpm':
            results['rpm_results'] = self._rpm_analysis(network, results)

        return results

    def _analyze_pairwise_direct(
        self,
        edge: Dict,
        quantiles: List[float]
    ) -> Dict:
        """Analyze a single pairwise comparison directly."""
        control, treatment = edge['raw_data']

        # Create analyzer for this pair
        config = IQMAConfig(
            quantiles=quantiles,
            n_bootstrap=500,
            use_random_effects=True,
            show_progress=False
        )

        analyzer = IPDQMA(config)
        results = analyzer.fit([edge['raw_data']])

        return {
            'treatment1': edge['treatment1'],
            'treatment2': edge['treatment_name2'],
            'n_samples': edge['n_treatment1'] + edge['n_treatment2'],
            'slope_p_value': results['slope_test']['p'],
            'lnvr_p_value': results['lnvr_test']['p'],
            'profile': results['profile']
        }

    def _node_split_analysis(
        self,
        treatment1: str,
        treatment2: str,
        data: Tuple[np.ndarray, np.ndarray],
        n1: int,
        n2: int
    ) -> Dict:
        """
        Perform node-splitting analysis for network data.

        This splits the network at a treatment node and analyzes
        the indirect comparison between two treatments via their common comparator.
        """
        # This is a simplified version
        # Full implementation would identify common comparators

        # For now, use direct analysis
        return self._analyze_pairwise_direct({
            'treatment1': treatment1,
            'treatment2': treatment2,
            'raw_data': data,
            'n_treatment1': n1,
            'n_treatment2': n2
        })

    def _check_transitivity(
        self,
        network: Dict,
        results: Dict
    ) -> Dict:
        """
        Check transitivity assumption across the network.

        For effects A vs B, B vs C, and A vs C, we check if:
        effect(A vs C) ≈ effect(A vs B) + effect(B vs C)

        at each quantile.
        """
        # Get all edges for each pair
        pairs = {}
        for edge_result in results['pairwise_results']:
            pair_key = frozenset([
                edge_result['treatment1'],
                edge_result['treatment2']
            ])
            pair_data = edge_result['results']

            # Store slope and lnVR for each quantile
            pairs[pair_key] = pair_data

        # Check all triplets for transitivity
        transitivity_violations = []

        treatments = network['treatments']
        for i, t1 in enumerate(treatments):
            for j, t2 in enumerate(treatments[i+1:], i):
                pair_key1 = frozenset([t1, t2])
                for k, t3 in enumerate(treatments[j+1:], j):
                    pair_key2 = frozenset([t2, t3])
                    pair_key3 = frozenset([t1, t3])

                    if pair_key1 in pairs and pair_key2 in pairs and pair_key3 in pairs:
                        # Check transitivity at each quantile
                        slopes = []
                        lnvrs = []

                        for q_idx, q in enumerate(results['quantiles']):
                            # Effects from pair_key1 at quantile q
                            profile1 = pairs[pair_key1]['profile']
                            slope1 = profile1[profile1['Quantile'] == q]['Effect']
                            lnvr1 = profile1[profile1['Quantile'] == q]['lnVR']

                            # Effects from pair_key2 at quantile q
                            profile2 = pairs[pair_key2]['profile']
                            slope2 = profile2[profile2['Quantile'] == q]['Effect']
                            lnvr2 = profile2[profile2['Quantile'] == q]['lnVR']

                            # Effects from pair_key3 at quantile q
                            profile3 = pairs[pair_key3]['profile']
                            slope3 = profile3[profile3['Quantile'] == q]['Effect']
                            lnvr3 = profile3[profile3['Quantile'] == q]['lnVR']

                            # Check: slope(A,C) ≈ slope(A,B) + slope(B,C)
                            expected_slope = slope1 + slope2
                            actual_slope = slope3

                            slopes.append((q, actual_slope, expected_slope))
                            lnvrs.append((q, lnvr1 + lnvr2, lnvr3))

                        # Check for violations
                        for q, actual, expected in slopes:
                            if abs(actual - expected) > 0.2:  # Threshold for violation
                                transitivity_violations.append({
                                    'quantile': q,
                                    't1': t1,
                                    't2': t2,
                                    't3': t3,
                                    'expected_slope': expected,
                                    'actual_slope': actual,
                                    'difference': actual - expected
                                })

        return {
            'n_triplets': len(slopes),
            'n_violations': len(transitivity_violations),
            'violations': transitivity_violations,
            'transitivity_met': len(transitivity_violations) == 0,
        }

    def _rpm_analysis(
        self,
        network: Dict,
        results: Dict
    ) -> Dict:
        """
        Rank Preserving Model for network consistency.

        This is a simplified implementation. The full RPM would involve
        complex optimization to preserve rank ordering of treatments.
        """
        # Calculate average ranks for each treatment across all quantiles
        treatment_effects = {}
        for treatment in network['treatments']:
            # Collect effects from edges involving this treatment
            effects = []
            for edge_result in results['pairwise_results']:
                if edge_result['treatment1'] == treatment:
                    effects.extend(edge_result['results']['profile']['Effect'].tolist())
                elif edge_result['treatment2'] == treatment:
                    # Note: this gives negative of effects since we want the effect of treatment vs comparator
                    effects.extend([-x for x in edge_result['results']['profile']['Effect'].tolist()])

            if effects:
                treatment_effects[treatment] = effects

        # Calculate ranks
        all_effects = []
        treatment_ranks = {}

        for treatment, effects in treatment_effects.items():
            all_effects.extend([(treatment, e) for e in effects])

        # Sort by effect size
        all_effects.sort(key=lambda x: x[1])

        # Assign ranks
        for rank, (treatment, _) in enumerate(all_effects):
            if treatment not in treatment_ranks:
                treatment_ranks[treatment] = []
            treatment_ranks[treatment].append(rank)

        # Calculate average rank
        avg_ranks = {t: np.mean(ranks) for t, ranks in treatment_ranks.items()}

        # Sort by average rank
        sorted_treatments = sorted(avg_ranks.items(), key=lambda x: x[1], reverse=True)

        return {
            'avg_ranks': avg_ranks,
            'league_table': [(t, rank, 1.0) for t, rank in sorted_treatments],
            'top_treatment': sorted_treatments[0][0],
            'bottom_treatment': sorted_treatments[-1][0],
            'rank_correlation': None  # Would need more complex analysis
        }

    def _generate_league_table(
        self,
        network: Dict,
        results:Dict
    ) -> pd.DataFrame:
        """
        Generate league table ranking treatments by quantile effects.

        Returns
        -------
        pandas.DataFrame
            League table with treatment rankings
        """
        treatments = network['treatments']

        # Create matrix: treatments x quantiles
        n_q = len(results['quantiles'])
        rank_matrix = np.zeros((n_q, len(treatments)))

        for q_idx, q in enumerate(results['quantiles']):
            # Get effects for each treatment at this quantile
            treatment_effects = {}

            for edge_result in results['pooled_results']:
                # Extract effects for both treatments
                profile = edge_result['results']['profile']
                for _, row in profile.iterrows():
                    t = row['Quantile']
                    if t == q:
                        effect = row['Effect']
                        tr = (edge_result['treatment1']
                              if edge_result['treatment1'] not in treatment_effects
                              else edge_result['treatment_name2'])
                        treatment_effects[tr] = effect

            # Calculate average effect for each treatment at this quantile
            avg_effects = {t: np.mean([v for v in vals])
                           for t, vals in treatment_effects.items()}

            # Rank treatments (higher effect = better, lower = worse)
            sorted_treatments = sorted(avg_effects.items(), key=lambda x: x[1], reverse=True)
            for rank, (t, _) in enumerate(sorted_treatments):
                rank_matrix[q_idx, list(treatments).index(t)] = rank

        # Create dataframe
        df = pd.DataFrame(
            rank_matrix,
            index=[f"Q{int(q*100)}%" for q in results['quantiles']],
            columns=treatments
        )

        return df

    def plot_network_graph(
        self,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Visualize the network structure with Plotly.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if self.network is None:
            raise ValueError("Build network first")

        import plotly.graph_objects as go
        import plotly.graph_objects as go

        # Create network graph
        fig = go.Figure()

        # Add nodes
        treatments = self.network['treatments']

        node_trace = go.Scatter(
            x=[1] * len(treatments),
            y=[1] * len(treatments),
            mode='markers+text',
            text=treatments,
            marker=dict(
                size=30,
                color='lightblue'
            )
        )

        # Add edges
        for edge in self.network['edges']:
            # Find indices
            idx1 = treatments.index(edge['treatment1'])
            idx2 = treatments.index(edge['treatment2'])

            fig.add_trace(go.Scatter(
                x=[idx1 + 1, idx2 + 1],
                y=[idx1 + 1, idx2 + 1],
                mode='lines+text',
                line=dict(color='gray', width=1),
                text=f"Edge: {edge['treatment_name']}",
                textposition='top center',
                textfont=dict(size=8),
                hovertemplate=f"{edge['study_id']}<br>HR: {edge['hazard_ratio']:.2f}"
            ))

        # Layout
        fig.update_layout(
            title="Network Graph: Treatment Comparison",
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=120, r=120, b=20, t=20, pad=10),
            plot_bgcolor='white',
            font=dict(size=10)
        )

        return fig

    def plot_network_heatmap(
        self,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot heatmap of effects across treatments and quantiles.

        Returns
 -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise ValueError("Run network analysis first")

        profile = self.results['profile']

        # Get unique treatments and quantiles
        treatments = list(set([
            r['treatment1'] for r in self.results['pairwise_results']
        ]))

        quantiles = self.results['quantiles']

        # Build effect matrix: treatments x quantiles
        effect_matrix = np.zeros((len(quantiles), len(treatments)))

        for q_idx, q in enumerate(quantiles):
            for t_idx, treatment in enumerate(treatments):
                effects = []

                for r in self.results['pairwise_results']:
                    if r['treatment1'] == treatment:
                        effects.append(r['results']['profile'][
                            r['results']['profile']['Quantile'] == q]['Effect'].values[0])
                    elif r['treatment2'] == treatment:
                        effects.append(-r['results']['profile'][
                            r['results']['profile']['Quantile'] == q]['Effect'].values[0])

                if effects:
                    effect_matrix[q_idx, t_idx] = np.mean(effects)

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(effect_matrix,
                           cmap='RdBu_r',
                           aspect='auto',
                           extent=[0, len(treatments), len(quantiles)],
                           vmin=-1, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(treatments)))
        ax.set_yticks(np.arange(len(quantiles)))
        ax.set_xticklabels(treatments)
        ax.set_yticklabels([f'{int(q*100)}%' for q in quantiles])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Effect Size')

        # Reference line at zero
        ax.axvline(len(treatments)/2, 0, color='gray', linestyle='--', linewidth=1)

        ax.set_title('Network Meta-Analysis: Effects Across Treatments and Quantiles')
        ax.set_xlabel('Treatment')
        ax.set_ylabel('Quantile')

        plt.tight_layout()
        return fig

    def plot_network_forest_plot(
        self,
        quantile_index: int = 0,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Create forest plot for network meta-analysis results.

        Parameters
        ----------
        quantile_index : int
            Index of quantile to plot (default: Q50)
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise ValueError("Run network analysis first")

        profile = self.results['profile']

        q_idx = quantile_index if quantile_index >= 0 else len(profile) // 2
        q = profile['Quantile'][q_idx]

        # Extract data for forest plot
        treatments = list(set([
            r['treatment1'] for r in self.results['pairwise_results']
        ]))

        n_treatments = len(treatments)

        # Extract effects for this quantile
        effects = {}
        se = {}
        study_names = []

        for r in self.results['pairwise_results']:
            t1 = r['treatment1']
            t2 = r['treatment2']

            # Extract effect at specified quantile
            effect = r['results']['profile'][
                r['results']['profile']['Quantile'] == q]['Effect'].values[0]

            # Get standard error
            # This is a simplified version - would need full extraction
            # of SE for each comparison

            # Create a unique key for each pair
            pair_key = (t1, t2)

            if pair_key not in effects:
                effects[pair_key] = []
                se[pair_key] = []

            effects[pair_key].append(effect)

            # We would need to extract the actual SE from the full results
            # For now, use a placeholder
            se[pair_key].append(0.1)

        # Create forest plot
        fig, ax = plt.subplots(figsize=figsize)

        y_positions = []
        y_labels = []
        effect_sizes = []
        cis = []

        y_pos = 0
        for t1 in treatments:
            for t2 in treatments:
                if t1 == t2:
                    continue

                pair_key = (t1, t2)
                if pair_key in effects:
                    effect_sizes.append(effects[pair_key])
                    y_positions.append(y_pos)
                    y_labels.append(f"{t1} vs {t2}")

                    # Create CI (using simplified SE)
                    ci_low = effects[pair_key] - 1.96 * se[pair_key][0]
            y_pos += 1

        ax.errorbar(
            effect_sizes,
            y_positions,
            xerr=1.96 * np.array([se[pair_key][0] for pair_key in effects if pair_key in se]),
            fmt='o',
            capsize=5,
            markersize=6,
            color='#00d4aa'
        )

        # Reference line at HR=1 (or effect=0)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Effect Size')
        ax.set_title(f'Network Forest Plot: Effect at {int(q*100)}% Quantile')
        ax.legend(['Indirect Comparisons'])
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(min(effect_sizes) - 0.1, max(effect_sizes) + 0.1)

        plt.tight_layout()
        return fig

    def plot_sucra_plot(
        self,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Create SUCRA (Surface Under Cumulative Rank Accumulation) plot.

        Shows how treatment rankings change across quantiles.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise ValueError("Run network analysis first")

        # Get RPM results
        rpm_results = self.results['rpm_results']

        # Extract league table data
        leagues = rpm_results['league_table']

        # Treatments and their ranks
        treatments = [t for t, _, _ in leagues]
        rank_data = {}

        for t, rank, score in leagues:
            if t not in rank_data:
                rank_data[t] = []

            rank_data[t].append(rank)

        # Create dataframe
        df = pd.DataFrame(rank_data, index=treatments).T

        # Create SUCRA plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each treatment's rank across quantiles
        for t in treatments:
            ax.plot(
                self.results['quantiles'],
                rank_data[t],
                marker='o-',
                label=t,
                linewidth=2
            )

        # Reference line (diagonal line where rank = index)
        ax.plot(self.results['quantiles'],
                 np.arange(len(self.results['quantiles'])),
                 'k--',
                 color='gray',
                 alpha=0.3)

        ax.set_xlabel('Quantile')
        ax.set_ylabel('Rank')
        ax.set_title('SUCRA Plot: Treatment Rankings Across Quantiles')
        ax.invert_yaxis()
        ax.set_ylim(len(self.results['quantiles']), 1, 0)

        # Add legend
        ax.legend(loc='best', fontsize='small')

        plt.tight_layout()
        return fig


# Example usage
def create_network_data():
    """Create example network meta-analysis data."""
    # 3-arm trial data
    studies = [
        {
            'study': 'TRIAL_A',
        'treatment': ['A', 'B', 'C'],
        'outcome': [10, 12, 15],
        'n': [50, 50, 50]
    },
        {
            'study': 'TRIAL_B',
        'treatment': ['A', 'B', 'C'],
        'outcome': [11, 13, 14],
        'n': [45, 48, 52]
    },
        {
            'study': 'TRIAL_C',
            'treatment': ['A', 'B', 'C'],
        'outcome': [9, 11, 16],
        'n': [48, 52, 47]
    }
    ]

    studies_data = []

    for study in studies:
        treatments = study['treatment']
        n_treatments = study['n']
        outcomes = study['outcome']
        n_samples = study['n']

        # Create raw data for each treatment
        for i, treatment in enumerate(treatments):
            # Generate individual participant data
            # For demonstration, we'll generate data around the mean
            mean_effect = 1.0 + i * 0.5  # Treatment effects differ
            effect_sd = 2.0
            error_sd = 1.5

            # Generate outcomes for this treatment
            # Add within-subject variability
            subject_data = []
            for j in range(n_samples):
                true_effect = mean_effect + np.random.normal(0, error_sd)
                subject_data.append(np.random.normal(true_effect, error_sd, n_samples))

        studies_data.append(study)

    return studies_data


if __name__ == "__main__":
    print("IPD-QMA Network Meta-Analysis")
    print("=" * 60)

    # Create network data
    studies_data = create_network_data()

    # Create network analyzer
    network_config = NetworkConfig(
        consistency_model='rpm',
        use_intransitivity_check=True
    )

    analyzer = IPDQMA(network_config)

    # Build network
    network = analyzer.build_network(studies_data, 'study', 'treatment', 'outcome')

    print(f"Network built: {network['n_treatments']} treatments, {network['n_edges']} comparisons")

    # Analyze network
    results = analyzer.analyze_network(network)

    print(f"\nNetwork Analysis Results:")
    print(f"  Transitivity violations: {results['transitivity']['n_violations']}")

    # Plot network graph
    if PLOTLY_AVAILABLE:
        try:
            fig = analyzer.plot_network_graph()
            fig.show()
            plt.close(fig)
        except:
            print("Install plotly for interactive network plots")

    # Plot heatmap
    fig = analyzer.plot_network_heatmap()
    plt.show()
    plt.close(fig)

    print("\nSUCRA Plot:")
    fig = analyzer.plot_sucra_plot()
    plt.show()
    plt.close(fig)

    print("\nLeague Table:")
    league_table = analyzer._generate_league_table(network, results)
    print(league_table.to_string())
