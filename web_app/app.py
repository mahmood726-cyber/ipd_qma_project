"""
IPD-QMA Web Application

A Streamlit-based interactive web application for IPD-QMA analysis.
Features:
- File upload (CSV, Excel)
- Interactive data input
- Real-time analysis
- Interactive visualizations
- Results export

Run with: streamlit run web_app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_qma import IPDQMA, IQMAConfig
from ipd_qma_advanced import IPDQMAAdvanced
from ipd_qma_validation import IPDQMAValidator
from ipd_qma_plots import IPDQMAPlotter

# Page configuration
st.set_page_config(
    page_title="IPD-QMA Web App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D4EFDF;
        border: 1px solid #27AE60;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FCF3CF;
        border: 1px solid #F39C12;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def load_example_data():
    """Load example datasets for demonstration."""
    datasets = {
        'Clinical Trial (Simulated)': {
            'description': 'Simulated clinical trial with heterogeneous treatment effects',
            'n_studies': 10,
            'n_per_group': 100,
            'effect_pattern': 'Heterogeneous (location + scale shift)'
        },
        'Educational Intervention': {
            'description': 'Educational study outcomes',
            'n_studies': 8,
            'n_per_group': 50,
            'effect_pattern': 'Mostly homogeneous'
        }
    }
    return datasets


def parse_uploaded_file(uploaded_file):
    """Parse uploaded CSV or Excel file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel."
        return df, None
    except Exception as e:
        return None, str(e)


def main():
    """Main application."""

    # Header
    st.markdown('<p class="main-header">IPD-QMA: Individual Participant Data Quantile Meta-Analysis</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #7F8C8D; margin-bottom: 2rem;">
    Detect heterogeneous treatment effects across patient severity distributions
    </p>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Quantile selection
        st.subheader("Quantiles")
        quantile_options = {
            'Standard (5 quantiles)': [0.1, 0.25, 0.5, 0.75, 0.9],
            'Extended (9 quantiles)': list(np.linspace(0.05, 0.95, 9)),
            'Fine (19 quantiles)': list(np.linspace(0.05, 0.95, 19)),
            'Custom': None
        }

        quantile_choice = st.selectbox('Select quantiles', list(quantile_options.keys()))

        if quantile_choice == 'Custom':
            quantiles = st.text_input('Enter quantiles (comma-separated, 0-1)', '0.1, 0.25, 0.5, 0.75, 0.9')
            try:
                quantiles = [float(q.strip()) for q in quantiles.split(',')]
                quantiles = sorted(quantiles)
            except:
                st.error("Invalid quantile format")
                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        else:
            quantiles = quantile_options[quantile_choice]

        # Bootstrap settings
        st.subheader("Bootstrap Settings")
        n_bootstrap = st.slider('Number of bootstrap samples', 100, 5000, 500, step=100)
        use_parallel = st.checkbox('Use parallel processing', value=True)
        n_workers = st.slider('Number of workers (0 = auto)', 0, 8, 0) if use_parallel else 1
        show_progress = st.checkbox('Show progress bars', value=True)

        # Analysis settings
        st.subheader("Analysis Settings")
        use_random_effects = st.checkbox('Use random-effects model', value=True)
        tau2_estimator = st.selectbox('Heterogeneity estimator', ['dl', 'pm'])
        confidence_level = st.selectbox('Confidence level', [0.90, 0.95, 0.99])

        # Reproducibility
        st.subheader("Reproducibility")
        random_seed = st.number_input('Random seed (0 for random)', 0, 10000, 42)

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(['📁 Data Input', '🔬 Analysis', '📊 Results', '⬇️ Export'])

    # Tab 1: Data Input
    with tab1:
        st.header("Upload or Enter Data")

        data_input_method = st.radio("Choose input method:", ['Upload File', 'Manual Input', 'Example Data'])

        studies_data = []

        if data_input_method == 'Upload File':
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Upload Instructions")
                st.markdown("""
                Upload your data in CSV or Excel format.

                **Required columns:**
                - `study_id`: Study identifier
                - `group`: 'control' or 'treatment'
                - `outcome`: Continuous outcome value

                **Example format:**
                | study_id | group | outcome |
                |----------|-------|---------|
                | 1 | control | 1.2 |
                | 1 | control | 1.5 |
                | 1 | treatment | 1.8 |
                | 2 | control | 0.9 |
                """)

            with col2:
                uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])

                if uploaded_file:
                    with st.spinner('Loading data...'):
                        df, error = parse_uploaded_file(uploaded_file)

                    if error:
                        st.error(f"Error loading file: {error}")
                    else:
                        st.success(f"Loaded {len(df)} rows")

                        # Show data preview
                        st.subheader("Data Preview")
                        st.dataframe(df.head(10))

                        # Check required columns
                        required_cols = ['study_id', 'group', 'outcome']
                        missing_cols = [c for c in required_cols if c not in df.columns]

                        if missing_cols:
                            st.error(f"Missing columns: {missing_cols}")
                        else:
                            # Process data
                            try:
                                for study_id in df['study_id'].unique():
                                    control = df[(df['study_id'] == study_id) & (df['group'] == 'control')]['outcome'].values
                                    treatment = df[(df['study_id'] == study_id) & (df['group'] == 'treatment')]['outcome'].values

                                    if len(control) > 0 and len(treatment) > 0:
                                        studies_data.append((control, treatment))

                                st.success(f"Processed {len(studies_data)} studies")
                                st.session_state.data_loaded = True
                            except Exception as e:
                                st.error(f"Error processing data: {e}")

        elif data_input_method == 'Manual Input':
            st.subheader("Manual Data Entry")

            n_studies = st.number_input('Number of studies', 1, 50, 3)

            for i in range(n_studies):
                with st.expander(f"Study {i+1}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Control Group**")
                        control_input = st.text_area(
                            f"Control outcomes (comma-separated)",
                            value="1.2, 1.5, 1.8, 2.1, 2.4",
                            key=f"control_{i}"
                        )

                    with col2:
                        st.markdown("**Treatment Group**")
                        treatment_input = st.text_area(
                            f"Treatment outcomes (comma-separated)",
                            value="1.8, 2.2, 2.5, 3.0, 3.5",
                            key=f"treatment_{i}"
                        )

                    try:
                        control = np.array([float(x.strip()) for x in control_input.split(',') if x.strip()])
                        treatment = np.array([float(x.strip()) for x in treatment_input.split(',') if x.strip()])

                        if len(control) > 0 and len(treatment) > 0:
                            studies_data.append((control, treatment))
                    except:
                        st.warning(f"Invalid data format for Study {i+1}")

            if studies_data:
                st.success(f"Loaded {len(studies_data)} studies")
                st.session_state.data_loaded = True

        elif data_input_method == 'Example Data':
            st.subheader("Example Datasets")

            example_datasets = load_example_data()

            for name, info in example_datasets.items():
                with st.expander(f"**{name}**"):
                    st.write(info['description'])
                    st.write(f"- Studies: {info['n_studies']}")
                    st.write(f"- Sample size per group: {info['n_per_group']}")
                    st.write(f"- Effect pattern: {info['effect_pattern']}")

                    if st.button(f"Load {name}", key=f"load_{name}"):
                        # Generate example data
                        np.random.seed(42)
                        for i in range(info['n_studies']):
                            base_scale = np.random.uniform(0.8, 1.2)
                            control = np.random.exponential(base_scale, info['n_per_group']) - 1
                            variance_multiplier = np.random.uniform(2.5, 3.5)
                            treatment = (np.random.exponential(base_scale, info['n_per_group']) - 1) * variance_multiplier

                            # Remove outliers
                            control = control[(control > -2) & (control < 5)]
                            treatment = treatment[(treatment > -2) & (treatment < 10)]

                            studies_data.append((control, treatment))

                        st.session_state.data_loaded = True
                        st.success(f"Loaded {len(studies_data)} example studies")
                        st.rerun()

        # Data validation
        if studies_data:
            st.subheader("Data Validation")

            with st.spinner('Validating data...'):
                validator = IPDQMAValidator(strict=False)
                validation_results = validator.validate_studies(studies_data)

            # Show validation summary
            overall = validation_results.get('overall', {})

            if overall.get('passed', False):
                st.markdown('<div class="success-box">All studies passed validation</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">Some studies have validation warnings</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Studies", overall.get('details', {}).get('n_studies', len(studies_data)))
            with col2:
                st.metric("Quality Score", f"{overall.get('score', 0):.1f}/100")
            with col3:
                st.metric("Status", "PASS" if overall.get('passed', False) else "REVIEW")

            # Show warnings
            all_warnings = []
            for study_name, result in validation_results.items():
                if study_name != 'overall' and result.warnings:
                    all_warnings.extend([f"{study_name}: {w}" for w in result.warnings[:2]])

            if all_warnings:
                with st.expander(f"View Warnings ({len(all_warnings)})"):
                    for warning in all_warnings[:10]:
                        st.warning(warning)

    # Tab 2: Analysis
    with tab2:
        st.header("Run IPD-QMA Analysis")

        if not st.session_state.data_loaded:
            st.info("Please load data in the Data Input tab first")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Analysis Configuration")
                st.markdown(f"""
                - **Quantiles**: {len(quantiles)} quantiles ({', '.join([f'{q:.0%}' for q in quantiles[:3]])}...)
                - **Bootstrap samples**: {n_bootstrap}
                - **Model**: {'Random-effects' if use_random_effects else 'Fixed-effect'}
                - **Confidence level**: {confidence_level:.0%}
                """)

            with col2:
                st.subheader("Actions")
                if st.button("▶ Run Analysis", type="primary", use_container_width=True):
                    # Create config
                    config = IQMAConfig(
                        quantiles=quantiles,
                        n_bootstrap=n_bootstrap,
                        confidence_level=confidence_level,
                        random_seed=random_seed if random_seed > 0 else None,
                        use_random_effects=use_random_effects,
                        tau2_estimator=tau2_estimator,
                        n_workers=n_workers if n_workers > 0 else None,
                        show_progress=show_progress
                    )

                    # Run analysis
                    with st.spinner('Running IPD-QMA analysis...'):
                        analyzer = IPDQMA(config)
                        results = analyzer.fit(studies_data)

                    # Store in session
                    st.session_state.analyzer = analyzer
                    st.session_state.results = results

                    st.success("Analysis complete!")
                    st.rerun()

    # Tab 3: Results
    with tab3:
        st.header("Analysis Results")

        if st.session_state.results is None:
            st.info("Please run analysis first")
        else:
            results = st.session_state.results

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Studies", results['n_studies'])
            with col2:
                st.metric("Model", results['model_type'].replace('_', ' ').title())
            with col3:
                slope_p = results['slope_test']['p']
                slope_status = "✓ Significant" if slope_p < 0.05 else "Not significant"
                st.metric("Slope Test", slope_status)
            with col4:
                lnvr_p = results['lnvr_test']['p']
                lnvr_status = "✓ Significant" if lnvr_p < 0.05 else "Not significant"
                st.metric("lnVR Test", lnvr_status)

            # Detailed results
            st.subheader("Slope Test (Heterogeneity)")
            slope = results['slope_test']
            st.markdown(f"""
            - **Estimate**: {slope['estimate']:.4f} (SE: {slope['se']:.4f})
            - **95% CI**: [{slope['ci_lower']:.4f}, {slope['ci_upper']:.4f}]
            - **P-value**: {slope['p']:.4f}
            - **I²**: {slope['i2']:.1f}%
            - **τ²**: {slope['tau2']:.4f}
            - **Interpretation**: {slope['interpretation']}
            """)

            st.subheader("Log Variance Ratio Test (Scale Shift)")
            lnvr = results['lnvr_test']
            st.markdown(f"""
            - **Estimate**: {lnvr['estimate']:.4f} (SE: {lnvr['se']:.4f})
            - **95% CI**: [{lnvr['ci_lower']:.4f}, {lnvr['ci_upper']:.4f}]
            - **P-value**: {lnvr['p']:.4f}
            - **I²**: {lnvr['i2']:.1f}%
            - **τ²**: {lnvr['tau2']:.4f}
            - **Interpretation**: {lnvr['interpretation']}
            """)

            # Quantile profile table
            st.subheader("Quantile Profile")
            profile_df = results['profile'].copy()
            profile_df['Quantile'] = profile_df['Quantile'].apply(lambda x: f"{x:.0%}")
            profile_df = profile_df.round(4)
            st.dataframe(profile_df, use_container_width=True)

            # Visualizations
            st.subheader("Visualizations")

            # Get analyzer
            analyzer = st.session_state.analyzer

            # Fan plot
            try:
                fig1 = analyzer.plot(figsize=(10, 5))
                st.pyplot(fig1)
                plt.close(fig1)
            except Exception as e:
                st.warning(f"Could not create fan plot: {e}")

            # Forest plot
            try:
                fig2 = analyzer.plot_forest(figsize=(10, 6))
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception as e:
                st.warning(f"Could not create forest plot: {e}")

    # Tab 4: Export
    with tab4:
        st.header("Export Results")

        if st.session_state.results is None:
            st.info("No results to export")
        else:
            st.subheader("Download Options")

            # Summary as CSV
            if st.session_state.analyzer:
                try:
                    summary = st.session_state.analyzer.summary()

                    # Convert to CSV
                    csv = summary.to_csv(index=False)

                    st.download_button(
                        label="📥 Download Summary (CSV)",
                        data=csv,
                        file_name="ipd_qma_summary.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.warning(f"Could not create CSV: {e}")

            # Results as JSON
            import json

            # Prepare results for JSON serialization
            results_json = {}
            for key, value in st.session_state.results.items():
                if key == 'profile':
                    results_json[key] = value.to_dict('records')
                elif key == 'study_details':
                    results_json[key] = [{k: v.tolist() if isinstance(v, np.ndarray) else v
                                         for k, v in s.items()} for s in value]
                elif key == 'config':
                    results_json[key] = {
                        'quantiles': value.quantiles,
                        'n_bootstrap': value.n_bootstrap,
                        'confidence_level': value.confidence_level,
                        'use_random_effects': value.use_random_effects
                    }
                else:
                    results_json[key] = value

            json_str = json.dumps(results_json, indent=2, default=str)

            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=json_str,
                file_name="ipd_qma_results.json",
                mime="application/json"
            )

            # Generate report
            if st.button("📄 Generate Analysis Report"):
                st.info("Report generation feature coming soon!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; font-size: 0.8rem;">
        IPD-QMA Web App v2.0 | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
