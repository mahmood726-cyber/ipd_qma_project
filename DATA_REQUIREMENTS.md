# IPD-QMA Data Requirements Guide

## What Data Do You Need?

### Required Data Structure

IPD-QMA requires **Individual Participant Data (IPD)** from multiple studies.

For each study, you need:
- **Control group outcomes**: Continuous outcome measurements from the control/placebo group
- **Treatment group outcomes**: Continuous outcome measurements from the treatment/intervention group

### Input Format

The data should be provided as a **list of tuples**, where each tuple contains two arrays:

```python
studies_data = [
    (control_outcomes_study1, treatment_outcomes_study1),
    (control_outcomes_study2, treatment_outcomes_study2),
    (control_outcomes_study3, treatment_outcomes_study3),
    # ... more studies
]
```

### Example Data Formats

#### Format 1: From CSV Files
```python
import pandas as pd

studies = []

# Study 1
df1 = pd.read_csv('study1.csv')
control1 = df1[df1['group'] == 'control']['outcome'].values
treatment1 = df1[df1['group'] == 'treatment']['outcome'].values
studies.append((control1, treatment1))

# Study 2
df2 = pd.read_csv('study2.csv')
control2 = df2[df2['group'] == 'control']['outcome'].values
treatment2 = df2[df2['group'] == 'treatment']['outcome'].values
studies.append((control2, treatment2))
```

#### Format 2: From R Data Packages
```python
# Load data from R packages using pyreadr or rpy2
import pyreadr

# Example: Reading from an RData file
result = pyreadr.read_r('study_data.RData')
control = result['control_outcomes']
treatment = result['treatment_outcomes']
studies.append((control, treatment))
```

#### Format 3: From Excel Files
```python
studies = []

for study_file in ['study1.xlsx', 'study2.xlsx', 'study3.xlsx']:
    df = pd.read_excel(study_file)
    control = df[df['arm'] == 'control']['score'].values
    treatment = df[df['arm'] == 'treatment']['score'].values
    studies.append((control, treatment))
```

### Data Quality Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Number of studies | 2 | 5+ |
| Sample size per group | 10 | 50+ |
| Outcome type | Continuous | Continuous |
| Missing data | 0% | <5% (imputed) |

### What IPD-QMA Analyzes

IPD-QMA detects **heterogeneous treatment effects** - where the treatment effect varies across the outcome distribution.

#### Examples of Suitable Research Questions:

1. **"Does the treatment work better for severely ill patients?"**
   - Outcome: Symptom severity score
   - Heterogeneity: Treatment effect larger at higher quantiles

2. **"Is there a variance shift between groups?"**
   - Outcome: Time to recovery
   - Heterogeneity: Treatment increases variability (scale shift)

3. **"Are treatment effects consistent across patient severity?"**
   - Outcome: Quality of life score
   - Heterogeneity: Flat fan plot = consistent effects

### Example: Complete Workflow

```python
import numpy as np
import pandas as pd
from ipd_qma import IPDQMA, IQMAConfig

# 1. Load your data
studies_data = []
for study_id in range(1, 11):
    df = pd.read_csv(f'study_{study_id}.csv')

    # Extract outcome measurements
    control = df[df['treatment'] == 0]['outcome'].values
    treatment = df[df['treatment'] == 1]['outcome'].values

    studies_data.append((control, treatment))

# 2. Configure analysis
config = IQMAConfig(
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
    n_bootstrap=500,
    use_random_effects=True
)

# 3. Run analysis
analyzer = IPDQMA(config)
results = analyzer.fit(studies_data)

# 4. View results
summary = analyzer.summary()

# 5. Generate plots
analyzer.plot()
analyzer.plot_forest()

# 6. Export results
analyzer.export_results('results.xlsx')
```

### Common Data Sources

| Source | Type | Access |
|--------|------|--------|
| **Clinical trials** | IPD | Trial registries, journal supplements |
| **CRAN packages** | Example datasets | `psych`, `MedCalc`, `meta` packages |
| **Zenodo/Figshare** | IPD repositories | Search by condition + "IPD" |
| **GitHub** | Research datasets | Search for specific domains |
| **ICMJE registries** | Clinical trial data | Via data request |

### Key References for Datasets

1. **CRAN Packages with IPD-like data:**
   - `meta` package: Contains datasets for meta-analysis
   - `metafor` package: Example datasets
   - `survival` package: Time-to-event data
   - `boot` package: Bootstrap example data

2. **Real IPD Resources:**
   - ClinicalStudyDataRequest.com
   - YODA Project (Yale)
   - dbGaP (NIH)
   - Vivli (harvard)

### Important Notes

1. **Continuous outcomes only** - IPD-QMA works with continuous measurements (scores, times, values)

2. **Skewed distributions are OK** - In fact, IPD-QMA excels at detecting effects in skewed data

3. **Small studies** - Minimum 10 per group, but power is limited

4. **Missing data** - Should be handled (imputed) before analysis

5. **Outliers** - Bootstrap is robust, but extreme outliers may need review
