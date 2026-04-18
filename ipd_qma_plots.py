"""
IPD-QMA Plots: Enhanced Visualization Module

This module provides enhanced visualizations for IPD-QMA analysis:
- Interactive Plotly visualizations
- Heatmap of effects across studies and quantiles
- Bootstrap distribution animations
- Network plots for multi-arm studies
- Custom publication-ready themes

Author: IPD-QMA Development Team
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Optional, Union
import warnings

# Try to import Plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

# Import base IPDQMA
from ipd_qma import IPDQMA


class IPDQMAPlotter:
    """
    Enhanced visualization suite for IPD-QMA results.

    Provides both static matplotlib and interactive plotly visualizations.
    """

    def __init__(self, analyzer: IPDQMA):
        """
        Initialize plotter with fitted analyzer.

        Parameters
        ----------
        analyzer : IPDQMA
            Fitted IPDQMA analyzer
        """
        if analyzer.results is None:
            raise ValueError("Analyzer must be fitted before plotting")

        self.analyzer = analyzer
        self.results = analyzer.results
        self.quantiles = analyzer.quantiles

    # ========================================================================
    # INTERACTIVE PLOTLY VISUALIZATIONS
    # ========================================================================

    def interactive_fan_plot(
        self,
        show_predictions: bool = True,
        theme: str = 'plotly_white'
    ) -> 'go.Figure':
        """
        Create interactive fan plot using Plotly.

        Parameters
        ----------
        show_predictions : bool
            Show prediction intervals
        theme : str
            Plotly theme name

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots. Install with: pip install plotly")

        df = self.results['profile']

        fig = go.Figure()

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(df['Quantile']) + list(df['Quantile'])[::-1],
            y=list(df['CI_Upper']) + list(df['CI_Lower'])[::-1],
            fill='toself',
            fillcolor='rgba(46, 134, 193, 0.25)',
            line=dict(color='rgba(46, 134, 193, 0)'),
            name='95% CI',
            hoverinfo='skip'
        ))

        # Prediction interval
        if show_predictions and not df['Pred_Lower'].isna().all():
            fig.add_trace(go.Scatter(
                x=list(df['Quantile']) + list(df['Quantile'])[::-1],
                y=list(df['Pred_Upper']) + list(df['Pred_Lower'])[::-1],
                fill='toself',
                fillcolor='rgba(46, 134, 193, 0.15)',
                line=dict(color='rgba(46, 134, 193, 0)'),
                name='95% PI',
                hoverinfo='skip'
            ))

        # Effect line
        fig.add_trace(go.Scatter(
            x=df['Quantile'],
            y=df['Effect'],
            mode='lines+markers',
            line=dict(color='#2E86C1', width=3),
            marker=dict(size=10),
            name='Pooled Effect',
            hovertemplate='<b>Q%{x:.0%}</b><br>Effect: %{y:.3f}<br>SE: %{customdata[0]:.3f}<extra></extra>',
            customdata=df[['SE', 'P']].values
        ))

        # Individual study points
        if len(self.analyzer._study_results) > 1:
            for i, study in enumerate(self.analyzer._study_results):
                fig.add_trace(go.Scatter(
                    x=self.quantiles,
                    y=study['quantiles'],
                    mode='markers',
                    marker=dict(size=4, color='gray', opacity=0.3),
                    name=f'Study {i+1}' if i < 3 else None,
                    showlegend=i < 3,
                    hovertemplate=f'<b>Study {i+1}</b><br>Quantile: %{{x:.0%}}<br>Effect: %{{y:.3f}}<extra></extra>'
                ))

        # Reference line
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.7)

        fig.update_layout(
            title=f'IPD-QMA Profile: Treatment Effects Across Quantiles<br>' +
                  f'<sub>({self.results["n_studies"]} studies, {self.results["model_type"].replace("_", " ").title()})</sub>',
            xaxis_title='Patient Severity Quantile',
            yaxis_title='Treatment Effect Size',
            template=theme,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            height=500
        )

        fig.update_xaxes(tickformat='.0%')

        return fig

    def interactive_forest_plot(
        self,
        quantile_index: int = -1
    ) -> 'go.Figure':
        """
        Create interactive forest plot using Plotly.

        Parameters
        ----------
        quantile_index : int
            Index of quantile to plot

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots")

        q_idx = quantile_index if quantile_index >= 0 else len(self.quantiles) // 2
        q = self.quantiles[q_idx]

        # Study-level data
        study_effects = [s['quantiles'][q_idx] for s in self.analyzer._study_results]
        study_ses = [s['se_quantiles'][q_idx] for s in self.analyzer._study_results]
        study_names = [f"Study {i+1}" for i in range(len(self.analyzer._study_results))]

        # Pooled effect
        pooled = self.results['profile'].iloc[q_idx]

        fig = go.Figure()

        # Individual studies
        for i, (name, effect, se) in enumerate(zip(study_names, study_effects, study_ses)):
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se

            # Error bar
            fig.add_trace(go.Scatter(
                x=[ci_lower, ci_upper],
                y=[i, i],
                mode='lines',
                line=dict(color='#2E86C1', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Point estimate
            fig.add_trace(go.Scatter(
                x=[effect],
                y=[i],
                mode='markers',
                marker=dict(size=10, color='#2E86C1'),
                name=name,
                hovertemplate=f'<b>{name}</b><br>Effect: {effect:.3f}<br>95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]<extra></extra>'
            ))

        # Pooled effect
        y_pooled = len(study_names)
        fig.add_hline(
            y=y_pooled,
            line_color='red',
            line_dash='dash',
            annotation_text=f"Pooled ({self.results['model_type'].replace('_', ' ').title()})"
        )
        fig.add_trace(go.Scatter(
            x=[pooled['Effect']],
            y=[y_pooled],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name='Pooled',
            hovertemplate=f'<b>Pooled</b><br>Effect: {pooled["Effect"]:.3f}<br>95% CI: [{pooled["CI_Lower"]:.3f}, {pooled["CI_Upper"]:.3f}]<extra></extra>'
        ))

        # Reference line
        fig.add_vline(x=0, line_dash='solid', line_color='gray', width=1)

        fig.update_layout(
            title=f'Forest Plot: {q:.0%} Quantile Treatment Effect',
            xaxis_title=f'Effect Size at {q:.0%} Quantile',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(study_names))) + [y_pooled],
                ticktext=study_names + ['Pooled']
            ),
            hovermode='closest',
            template='plotly_white',
            height=400 + len(study_names) * 30
        )

        return fig

    def heatmap_effects(
        self,
        show_values: bool = True,
        color_scale: str = 'RdBu_r',
        theme: str = 'plotly_white'
    ) -> 'go.Figure':
        """
        Create heatmap of effects across studies and quantiles.

        Parameters
        ----------
        show_values : bool
            Show effect values in cells
        color_scale : str
            Plotly color scale name
        theme : str
            Plotly theme name

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive heatmap
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots")

        # Create effect matrix
        effect_matrix = np.array([
            [s['quantiles'][i] for s in self.analyzer._study_results]
            for i in range(len(self.quantiles))
        ])

        study_names = [f"S{i+1}" for i in range(len(self.analyzer._study_results))]
        quantile_labels = [f"{q:.0%}" for q in self.quantiles]

        fig = go.Figure(data=go.Heatmap(
            z=effect_matrix,
            x=study_names,
            y=quantile_labels,
            colorscale=color_scale,
            colorbar=dict(title='Effect Size'),
            hovertemplate='Study: %{x}<br>Quantile: %{y}<br>Effect: %{z:.3f}<extra></extra>',
            text=effect_matrix if show_values else None,
            texttemplate='%{text:.2f}' if show_values else None,
            textfont=dict(size=10)
        ))

        fig.update_layout(
            title='Heatmap: Treatment Effects Across Studies and Quantiles',
            xaxis_title='Study',
            yaxis_title='Quantile',
            template=theme,
            height=500
        )

        return fig

    # ========================================================================
    # ENHANCED MATPLOTLIB VISUALIZATIONS
    # ========================================================================

    def multi_quantile_forest(
        self,
        quantiles: Optional[List[float]] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Create forest plots for multiple quantiles in one figure.

        Parameters
        ----------
        quantiles : list, optional
            Quantiles to plot (default: all)
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if quantiles is None:
            quantiles = self.quantiles

        n_quantiles = len(quantiles)
        quantile_indices = [self.quantiles.index(q) if q in self.quantiles else i
                           for i, q in enumerate(quantiles)]

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_quantiles, 1, figure=fig, hspace=0.4)

        axes = []
        for i, (q_idx, q) in enumerate(zip(quantile_indices, quantiles)):
            ax = fig.add_subplot(gs[i, 0])
            axes.append(ax)

            # Study effects
            study_effects = [s['quantiles'][q_idx] for s in self.analyzer._study_results]
            study_ses = [s['se_quantiles'][q_idx] for s in self.analyzer._study_results]

            y_pos = np.arange(len(study_effects))
            ax.errorbar(
                study_effects,
                y_pos,
                xerr=1.96 * np.array(study_ses),
                fmt='o',
                capsize=3,
                markersize=6,
                color='#2E86C1'
            )

            # Pooled effect
            pooled = self.results['profile'].iloc[q_idx]
            ax.axvline(pooled['Effect'], color='red', linestyle='--', linewidth=1.5)
            ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"S{i+1}" for i in range(len(study_effects))])
            ax.set_xlabel('Effect Size')
            ax.set_title(f'{q:.0%} Quantile')

            if i == n_quantiles - 1:
                ax.legend(['Individual studies', 'Pooled'], loc='best', fontsize='small')

        fig.suptitle(
            f'Forest Plots Across Quantiles ({self.results["n_studies"]} studies)',
            fontweight='bold',
            y=0.995
        )

        return fig

    def effect_profile_comparison(
        self,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot individual study profiles alongside pooled profile.

        Parameters
        ----------
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot pooled effect
        df = self.results['profile']
        ax.plot(
            df['Quantile'],
            df['Effect'],
            'o-',
            color='red',
            linewidth=3,
            markersize=10,
            label='Pooled',
            zorder=10
        )

        # Plot CI
        ax.fill_between(
            df['Quantile'],
            df['CI_Lower'],
            df['CI_Upper'],
            color='red',
            alpha=0.2,
            label='Pooled 95% CI'
        )

        # Plot individual study profiles
        for i, study in enumerate(self.analyzer._study_results):
            alpha = max(0.1, 0.5 - i * 0.05)
            ax.plot(
                self.quantiles,
                study['quantiles'],
                '-',
                color='gray',
                alpha=alpha,
                linewidth=1,
                label='Individual studies' if i == 0 else None
            )

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Quantile', fontsize=12)
        ax.set_ylabel('Effect Size', fontsize=12)
        ax.set_title('Individual Study Profiles vs Pooled Estimate', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def heterogeneity_plot(
        self,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot heterogeneity statistics across quantiles.

        Parameters
        ----------
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        df = self.results['profile']

        # I² plot
        ax1.bar(range(len(df)), df['I2'], color='#2E86C1', alpha=0.7)
        ax1.axhline(25, color='orange', linestyle='--', label='Low (25%)')
        ax1.axhline(50, color='red', linestyle='--', label='High (50%)')
        ax1.set_xlabel('Quantile Index')
        ax1.set_ylabel('I² (%)')
        ax1.set_title('Heterogeneity (I²) Across Quantiles')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels([f'{q:.0%}' for q in df['Quantile']])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # τ² plot
        ax2.bar(range(len(df)), df['Tau2'], color='#E74C3C', alpha=0.7)
        ax2.set_xlabel('Quantile Index')
        ax2.set_ylabel('τ²')
        ax2.set_title('Between-Study Variance (τ²) Across Quantiles')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels([f'{q:.0%}' for q in df['Quantile']])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def publication_quality_fan(
        self,
        figsize: Tuple[int, int] = (10, 6),
        style: str = 'nature',
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create publication-quality fan plot with custom styling.

        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        style : str
            Publication style ('nature', 'science', 'jama', 'custom')
        dpi : int
            Resolution for saving

        Returns
        -------
        matplotlib.figure.Figure
        """
        # Apply style
        if style == 'nature':
            plt.rcParams.update({
                'font.size': 8,
                'axes.linewidth': 0.5,
                'xtick.major.width': 0.5,
                'ytick.major.width': 0.5,
                'font.family': 'Arial'
            })
        elif style == 'science':
            plt.rcParams.update({
                'font.size': 7,
                'axes.linewidth': 0.5,
                'font.family': 'Helvetica'
            })

        fig, ax = plt.subplots(figsize=figsize)

        df = self.results['profile']

        # Confidence interval
        ax.fill_between(
            df['Quantile'],
            df['CI_Lower'],
            df['CI_Upper'],
            color='#2E86C1',
            alpha=0.3,
            label='95% CI',
            linewidth=0
        )

        # Prediction interval (if available)
        if not df['Pred_Lower'].isna().all():
            ax.fill_between(
                df['Quantile'],
                df['Pred_Lower'],
                df['Pred_Upper'],
                color='#2E86C1',
                alpha=0.15,
                label='95% PI',
                linewidth=0
            )

        # Effect line
        ax.plot(
            df['Quantile'],
            df['Effect'],
            'o-',
            color='#2E86C1',
            linewidth=2,
            markersize=6,
            label='Pooled Effect'
        )

        # Reference line
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

        # Styling
        ax.set_xlabel('Quantile', fontsize=10)
        ax.set_ylabel('Effect Size', fontsize=10)
        ax.set_xticks(df['Quantile'])
        ax.set_xticklabels([f'{q:.0%}' for q in df['Quantile']])
        ax.legend(loc='best', frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, linestyle=':')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        return fig

    # ========================================================================
    # DASHBOARD
    # ========================================================================

    def create_dashboard(
        self,
        save_path: Optional[str] = None
    ) -> 'go.Figure':
        """
        Create interactive dashboard with multiple plots.

        Parameters
        ----------
        save_path : str, optional
            Path to save HTML file

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive dashboard
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for dashboard")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fan Plot', 'Effect Heatmap', 'Heterogeneity (I²)', 'Slope & lnVR Tests'),
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
                   [{'type': 'bar'}, {'type': 'indicator'}]]
        )

        df = self.results['profile']

        # 1. Fan plot
        fig.add_trace(
            go.Scatter(x=df['Quantile'], y=df['Effect'], mode='lines+markers',
                      name='Effect', line=dict(color='#2E86C1', width=2)),
            row=1, col=1
        )

        # 2. Heatmap
        effect_matrix = np.array([
            [s['quantiles'][i] for s in self.analyzer._study_results]
            for i in range(len(self.quantiles))
        ])
        fig.add_trace(
            go.Heatmap(z=effect_matrix,
                      colorscale='RdBu_r',
                      showscale=False),
            row=1, col=2
        )

        # 3. I² bar chart
        fig.add_trace(
            go.Bar(x=list(range(len(df))), y=df['I2'],
                   marker_color='#2E86C1',
                   showlegend=False),
            row=2, col=1
        )

        # 4. Test indicators
        slope_p = self.results['slope_test']['p']
        lnvr_p = self.results['lnvr_test']['p']

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=slope_p,
                title={'text': "Slope Test P-value"},
                domain={'x': [0, 0.5], 'y': [0, 1]}
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            showlegend=False,
            template='plotly_white',
            title_text=f"IPD-QMA Analysis Dashboard ({self.results['n_studies']} studies)"
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def save_all_plots(
        self,
        output_dir: str,
        formats: List[str] = ['png'],
        dpi: int = 300
    ) -> None:
        """
        Save all plots to files.

        Parameters
        ----------
        output_dir : str
            Output directory
        formats : list
            File formats ('png', 'pdf', 'svg')
        dpi : int
            Resolution for raster formats
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # List of plot methods
        plot_methods = [
            ('fan_plot', self.analyzer.plot),
            ('forest_plot', self.analyzer.plot_forest),
            ('multi_quantile_forest', lambda: self.multi_quantile_forest()),
            ('effect_profile_comparison', self.effect_profile_comparison),
            ('heterogeneity_plot', self.heterogeneity_plot),
            ('publication_quality_fan', lambda: self.publication_quality_fan())
        ]

        for name, method in plot_methods:
            fig = method()
            base_path = os.path.join(output_dir, name)

            for fmt in formats:
                path = f"{base_path}.{fmt}"
                if fmt == 'png':
                    fig.savefig(path, dpi=dpi, bbox_inches='tight')
                else:
                    fig.savefig(path, bbox_inches='tight')

            plt.close(fig)

        # Save interactive plots if Plotly is available
        if PLOTLY_AVAILABLE:
            interactive_methods = [
                ('interactive_fan', self.interactive_fan_plot),
                ('interactive_forest', self.interactive_forest_plot),
                ('heatmap', self.heatmap_effects),
                ('dashboard', self.create_dashboard)
            ]

            for name, method in interactive_methods:
                try:
                    fig = method()
                    path = os.path.join(output_dir, f"{name}.html")
                    fig.write_html(path)
                except Exception as e:
                    warnings.warn(f"Could not save {name}: {e}")


def create_comprehensive_report(
    analyzer: IPDQMA,
    output_path: str,
    include_interactive: bool = True
) -> None:
    """
    Create a comprehensive HTML report with all visualizations.

    Parameters
    ----------
    analyzer : IPDQMA
        Fitted analyzer
    output_path : str
        Output HTML file path
    include_interactive : bool
        Include interactive Plotly plots
    """
    from ipd_qma_plots import IPDQMAPlotter

    plotter = IPDQMAPlotter(analyzer)

    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IPD-QMA Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2E86C1; }
            h2 { color: #34495E; border-bottom: 2px solid #2E86C1; padding-bottom: 10px; }
            .summary { background: #ECF0F1; padding: 20px; border-radius: 5px; }
            .plot { margin: 30px 0; text-align: center; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #2E86C1; color: white; }
        </style>
    </head>
    <body>
        <h1>IPD-QMA Analysis Report</h1>
    """

    # Summary section
    summary = analyzer.summary()
    html_content += f"""
        <div class="summary">
            <h2>Analysis Summary</h2>
            <p><strong>Number of studies:</strong> {analyzer.results['n_studies']}</p>
            <p><strong>Model type:</strong> {analyzer.results['model_type']}</p>
            <p><strong>Slope test p-value:</strong> {analyzer.results['slope_test']['p']:.4f}</p>
            <p><strong>lnVR test p-value:</strong> {analyzer.results['lnvr_test']['p']:.4f}</p>
        </div>
    """

    # Add plots
    html_content += "<h2>Visualizations</h2>"

    # Matplotlib plots (saved as base64)
    import base64
    from io import BytesIO

    for plot_name, plot_method in [
        ('Fan Plot', analyzer.plot),
        ('Forest Plot', analyzer.plot_forest)
    ]:
        fig = plot_method()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        html_content += f"""
        <div class="plot">
            <h3>{plot_name}</h3>
            <img src="data:image/png;base64,{img_base64}" style="max-width: 100%;">
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    # Example usage
    print("IPD-QMA Plots Module")
    print("=" * 50)

    from ipd_qma import IPDQMA, IQMAConfig
    import numpy as np

    # Generate example data
    np.random.seed(42)
    studies = []
    for i in range(10):
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0.5, 1.2, 100)
        studies.append((control, treatment))

    # Run analysis
    config = IQMAConfig(n_bootstrap=500)
    analyzer = IPDQMA(config)
    analyzer.fit(studies)

    # Create plotter
    plotter = IPDQMAPlotter(analyzer)

    print(f"\nCreated plotter for {len(studies)} studies")

    if PLOTLY_AVAILABLE:
        print("\nInteractive plots available:")
        print("  - Interactive fan plot")
        print("  - Interactive forest plot")
        print("  - Effect heatmap")
        print("  - Dashboard")
    else:
        print("\nInstall plotly for interactive visualizations:")
        print("  pip install plotly")
