"""
IPD-QMA Validation: Data Validation and Quality Assessment Module

This module provides comprehensive data validation and quality assessment:
- Distribution normality tests
- Outlier detection algorithms
- Small sample size warnings
- Data quality scoring
- Missing data patterns analysis

Author: IPD-QMA Development Team
Version: 1.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    warnings: List[str]
    errors: List[str]
    score: float  # 0-100
    details: Dict


class IPDQMAValidator:
    """
    Data validation and quality assessment for IPD-QMA.

    Provides comprehensive validation checks before running analysis.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Parameters
        ----------
        strict : bool
            If True, treat warnings as errors
        """
        self.strict = strict
        self.validation_results = {}

    def validate_study(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        study_name: str = "Study"
    ) -> ValidationResult:
        """
        Comprehensive validation of a single study.

        Parameters
        ----------
        control : array-like
            Control group outcomes
        treatment : array-like
            Treatment group outcomes
        study_name : str
            Name of the study for reporting

        Returns
        -------
        ValidationResult
            Validation results with warnings and errors
        """
        control = np.asarray(control)
        treatment = np.asarray(treatment)

        warnings_list = []
        errors_list = []
        details = {}

        # Basic checks
        basic_result = self._check_basic_requirements(control, treatment)
        errors_list.extend(basic_result['errors'])
        warnings_list.extend(basic_result['warnings'])
        details['basic'] = basic_result['details']

        if not basic_result['passed'] and self.strict:
            return ValidationResult(
                passed=False,
                warnings=warnings_list,
                errors=errors_list,
                score=0,
                details=details
            )

        # Distribution tests
        dist_result = self._test_distributions(control, treatment)
        warnings_list.extend(dist_result['warnings'])
        details['distributions'] = dist_result['details']

        # Outlier detection
        outlier_result = self._detect_outliers(control, treatment)
        warnings_list.extend(outlier_result['warnings'])
        details['outliers'] = outlier_result['details']

        # Sample size assessment
        size_result = self._assess_sample_size(control, treatment)
        warnings_list.extend(size_result['warnings'])
        details['sample_size'] = size_result['details']

        # Calculate quality score
        score = self._calculate_quality_score(details)

        passed = len(errors_list) == 0

        return ValidationResult(
            passed=passed,
            warnings=warnings_list,
            errors=errors_list,
            score=score,
            details=details
        )

    def validate_studies(
        self,
        studies_data: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple studies.

        Parameters
        ----------
        studies_data : list of tuples
            List of (control, treatment) pairs

        Returns
        -------
        dict
            Dictionary mapping study index to validation results
        """
        results = {}
        for i, (control, treatment) in enumerate(studies_data):
            results[f"Study_{i+1}"] = self.validate_study(control, treatment)

        # Overall validation
        all_passed = all(r.passed for r in results.values())
        avg_score = np.mean([r.score for r in results.values()])

        results['overall'] = ValidationResult(
            passed=all_passed,
            warnings=[f"Average quality score: {avg_score:.1f}/100"],
            errors=[] if all_passed else ["Some studies failed validation"],
            score=avg_score,
            details={'n_studies': len(studies_data)}
        )

        self.validation_results = results
        return results

    def _check_basic_requirements(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> Dict:
        """Check basic data requirements."""
        errors = []
        warnings = []
        details = {}

        # Check for NaN
        if np.any(np.isnan(control)):
            errors.append(f"Control group has {np.sum(np.isnan(control))} NaN values")
        if np.any(np.isnan(treatment)):
            errors.append(f"Treatment group has {np.sum(np.isnan(treatment))} NaN values")

        # Check for infinite values
        if np.any(np.isinf(control)):
            errors.append("Control group has infinite values")
        if np.any(np.isinf(treatment)):
            errors.append("Treatment group has infinite values")

        # Check data types
        if not np.issubdtype(control.dtype, np.number):
            errors.append("Control group is not numeric")
        if not np.issubdtype(treatment.dtype, np.number):
            errors.append("Treatment group is not numeric")

        # Check variance
        if np.var(control, ddof=1) == 0:
            errors.append("Control group has zero variance")
        if np.var(treatment, ddof=1) == 0:
            errors.append("Treatment group has zero variance")

        details = {
            'n_control': len(control),
            'n_treatment': len(treatment),
            'has_nan': np.any(np.isnan(control)) or np.any(np.isnan(treatment)),
            'has_inf': np.any(np.isinf(control)) or np.any(np.isinf(treatment)),
            'var_control': np.var(control, ddof=1),
            'var_treatment': np.var(treatment, ddof=1)
        }

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'details': details
        }

    def _test_distributions(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> Dict:
        """Test distribution properties."""
        warnings = []
        details = {}

        # Shapiro-Wilk test for normality (if sample size <= 5000)
        if len(control) <= 5000:
            stat_c, p_c = stats.shapiro(control)
            details['shapiro_control'] = {'statistic': stat_c, 'p_value': p_c}

            if p_c < 0.001:
                warnings.append(f"Control group significantly non-normal (p={p_c:.2e})")

        if len(treatment) <= 5000:
            stat_t, p_t = stats.shapiro(treatment)
            details['shapiro_treatment'] = {'statistic': stat_t, 'p_value': p_t}

            if p_t < 0.001:
                warnings.append(f"Treatment group significantly non-normal (p={p_t:.2e})")

        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, ks_p = stats.ks_2samp(control, treatment)
        details['ks_test'] = {'statistic': ks_stat, 'p_value': ks_p}

        if ks_p < 0.001:
            warnings.append(f"Distributions significantly differ (KS p={ks_p:.2e})")

        # Skewness and kurtosis
        from scipy.stats import skew, kurtosis
        details['skewness_control'] = skew(control)
        details['skewness_treatment'] = skew(treatment)
        details['kurtosis_control'] = kurtosis(control)
        details['kurtosis_treatment'] = kurtosis(treatment)

        # Check for extreme skewness
        if abs(skew(control)) > 2:
            warnings.append(f"Control group highly skewed (skew={skew(control):.2f})")
        if abs(skew(treatment)) > 2:
            warnings.append(f"Treatment group highly skewed (skew={skew(treatment):.2f})")

        return {
            'warnings': warnings,
            'details': details
        }

    def _detect_outliers(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> Dict:
        """Detect outliers using multiple methods."""
        warnings = []
        details = {}

        # Method 1: IQR method
        def detect_outliers_iqr(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return np.where((data < lower) | (data > upper))[0]

        outliers_c_iqr = detect_outliers_iqr(control)
        outliers_t_iqr = detect_outliers_iqr(treatment)

        details['outliers_iqr_control'] = len(outliers_c_iqr)
        details['outliers_iqr_treatment'] = len(outliers_t_iqr)

        if len(outliers_c_iqr) > len(control) * 0.1:
            warnings.append(f"Control group has many outliers ({len(outliers_c_iqr)}/{len(control)})")
        if len(outliers_t_iqr) > len(treatment) * 0.1:
            warnings.append(f"Treatment group has many outliers ({len(outliers_t_iqr)}/{len(treatment)})")

        # Method 2: Z-score method
        def detect_outliers_zscore(data, threshold=3):
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return np.where(z_scores > threshold)[0]

        outliers_c_z = detect_outliers_zscore(control)
        outliers_t_z = detect_outliers_zscore(treatment)

        details['outliers_zscore_control'] = len(outliers_c_z)
        details['outliers_zscore_treatment'] = len(outliers_t_z)

        # Method 3: Median Absolute Deviation (MAD)
        def detect_outliers_mad(data, threshold=3.5):
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.where(np.abs(modified_z_scores) > threshold)[0]

        outliers_c_mad = detect_outliers_mad(control)
        outliers_t_mad = detect_outliers_mad(treatment)

        details['outliers_mad_control'] = len(outliers_c_mad)
        details['outliers_mad_treatment'] = len(outliers_t_mad)

        return {
            'warnings': warnings,
            'details': details
        }

    def _assess_sample_size(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> Dict:
        """Assess sample size adequacy."""
        warnings = []
        details = {}

        n_c = len(control)
        n_t = len(treatment)

        # Minimum sample sizes
        if n_c < 10:
            warnings.append(f"Control group very small (n={n_c})")
        elif n_c < 30:
            warnings.append(f"Control group small (n={n_c}), bootstrap may be unstable")

        if n_t < 10:
            warnings.append(f"Treatment group very small (n={n_t})")
        elif n_t < 30:
            warnings.append(f"Treatment group small (n={n_t}), bootstrap may be unstable")

        # Sample size imbalance
        ratio = max(n_c, n_t) / min(n_c, n_t)
        if ratio > 3:
            warnings.append(f"Sample size imbalance detected (ratio={ratio:.2f})")

        # Required sample size for power
        # Assuming effect size d=0.5, alpha=0.05, power=0.80
        from scipy.stats import norm
        z_alpha = norm.ppf(0.975)
        z_beta = norm.ppf(0.80)
        effect_size = 0.5
        required_n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        details['required_n_for_80_power'] = int(np.ceil(required_n))
        details['adequate_power'] = min(n_c, n_t) >= required_n

        if not details['adequate_power']:
            warnings.append(f"Sample size may be underpowered (need n>={int(required_n)})")

        return {
            'warnings': warnings,
            'details': details
        }

    def _calculate_quality_score(self, details: Dict) -> float:
        """
        Calculate overall quality score (0-100).

        Scoring:
        - Basic requirements: 40 points (all or nothing)
        - Distribution normality: 20 points
        - Outlier detection: 20 points
        - Sample size: 20 points
        """
        score = 0

        # Basic requirements (40 points)
        if details.get('basic', {}).get('has_nan', True) == False and \
           details.get('basic', {}).get('has_inf', True) == False:
            score += 40

        # Distribution normality (20 points)
        if 'distributions' in details:
            dist = details['distributions']
            # Penalize extreme non-normality
            shapiro_p_c = dist.get('shapiro_control', {}).get('p_value', 1)
            shapiro_p_t = dist.get('shapiro_treatment', {}).get('p_value', 1)

            if shapiro_p_c > 0.05 and shapiro_p_t > 0.05:
                score += 20  # Both normal
            elif shapiro_p_c > 0.001 and shapiro_p_t > 0.001:
                score += 10  # Not significantly non-normal
            # Otherwise no points

        # Outlier detection (20 points)
        if 'outliers' in details:
            out = details['outliers']
            n_c = details['basic']['n_control']
            n_t = details['basic']['n_treatment']

            outlier_pct_c = out.get('outliers_iqr_control', 0) / n_c
            outlier_pct_t = out.get('outliers_iqr_treatment', 0) / n_t

            avg_outlier_pct = (outlier_pct_c + outlier_pct_t) / 2

            if avg_outlier_pct < 0.05:
                score += 20  # Few outliers
            elif avg_outlier_pct < 0.10:
                score += 10  # Moderate outliers
            # Otherwise no points

        # Sample size (20 points)
        if 'sample_size' in details:
            size = details['sample_size']
            if size.get('adequate_power', False):
                score += 20  # Adequate power
            elif min(details['basic']['n_control'], details['basic']['n_treatment']) >= 30:
                score += 10  # Moderate sample size
            # Otherwise no points

        return min(100, max(0, score))

    def suggest_improvements(
        self,
        result: ValidationResult
    ) -> List[str]:
        """
        Suggest improvements based on validation results.

        Parameters
        ----------
        result : ValidationResult
            Validation result

        Returns
        -------
        list of str
            Suggested improvements
        """
        suggestions = []

        # Check for NaN/inf
        if result.details.get('basic', {}).get('has_nan', False):
            suggestions.append("Remove or impute missing values (NaN)")

        # Check sample size
        if 'sample_size' in result.details:
            size = result.details['sample_size']
            if not size.get('adequate_power', True):
                required = size.get('required_n_for_80_power', 0)
                suggestions.append(f"Increase sample size to at least {required} per group for adequate power")

        # Check for outliers
        if 'outliers' in result.details:
            out = result.details['outliers']
            if out.get('outliers_iqr_control', 0) > 5 or \
               out.get('outliers_iqr_treatment', 0) > 5:
                suggestions.append("Review and potentially remove extreme outliers")

        # Check distributions
        if 'distributions' in result.details:
            dist = result.details['distributions']
            if dist.get('skewness_control', 0) < -2 or dist.get('skewness_control', 0) > 2:
                suggestions.append("Consider transforming control group data (e.g., log transform)")
            if dist.get('skewness_treatment', 0) < -2 or dist.get('skewness_treatment', 0) > 2:
                suggestions.append("Consider transforming treatment group data (e.g., log transform)")

        return suggestions

    def generate_validation_report(
        self,
        studies_data: List[Tuple[np.ndarray, np.ndarray]]
    ) -> str:
        """
        Generate a text validation report.

        Parameters
        ----------
        studies_data : list of tuples
            Study data

        Returns
        -------
        str
            Validation report
        """
        results = self.validate_studies(studies_data)

        report = []
        report.append("=" * 70)
        report.append("IPD-QMA DATA VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")

        for study_name, result in results.items():
            if study_name == 'overall':
                continue

            report.append(f"{study_name}:")
            report.append(f"  Status: {'PASS' if result.passed else 'FAIL'}")
            report.append(f"  Quality Score: {result.score:.1f}/100")

            if result.warnings:
                report.append(f"  Warnings ({len(result.warnings)}):")
                for w in result.warnings[:5]:  # Limit to 5
                    report.append(f"    - {w}")
                if len(result.warnings) > 5:
                    report.append(f"    ... and {len(result.warnings) - 5} more")

            if result.errors:
                report.append(f"  Errors ({len(result.errors)}):")
                for e in result.errors:
                    report.append(f"    - {e}")

            # Suggestions
            suggestions = self.suggest_improvements(result)
            if suggestions:
                report.append(f"  Suggestions:")
                for s in suggestions:
                    report.append(f"    - {s}")

            report.append("")

        # Overall summary
        overall = results['overall']
        report.append("=" * 70)
        report.append("OVERALL SUMMARY")
        report.append("=" * 70)
        report.append(f"  Total Studies: {overall.details.get('n_studies', 0)}")
        report.append(f"  Average Quality Score: {overall.score:.1f}/100")
        report.append(f"  Status: {'ALL PASSED' if overall.passed else 'SOME FAILED'}")
        report.append("")

        return "\n".join(report)


def quick_validate(
    control: np.ndarray,
    treatment: np.ndarray
) -> Tuple[bool, List[str]]:
    """
    Quick validation check.

    Parameters
    ----------
    control : array-like
        Control group data
    treatment : array-like
        Treatment group data

    Returns
    -------
    tuple
        (passed, warnings)
    """
    validator = IPDQMAValidator(strict=False)
    result = validator.validate_study(control, treatment)

    return result.passed, result.warnings


def clean_data(
    control: np.ndarray,
    treatment: np.ndarray,
    remove_outliers: bool = True,
    impute_method: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean data by removing NaN, infinite values, and optionally outliers.

    Parameters
    ----------
    control : array-like
        Control group data
    treatment : array-like
        Treatment group data
    remove_outliers : bool
        Remove outliers using MAD method
    impute_method : str
        Not used currently (for future imputation)

    Returns
    -------
    tuple
        (cleaned_control, cleaned_treatment)
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    # Remove NaN and infinite
    mask_c = ~np.isnan(control) & ~np.isinf(control)
    mask_t = ~np.isnan(treatment) & ~np.isinf(treatment)

    control_clean = control[mask_c]
    treatment_clean = treatment[mask_t]

    # Remove outliers if requested
    if remove_outliers:
        def remove_outliers_mad(data):
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return data[np.abs(modified_z_scores) < 3.5]

        control_clean = remove_outliers_mad(control_clean)
        treatment_clean = remove_outliers_mad(treatment_clean)

    return control_clean, treatment_clean


if __name__ == "__main__":
    # Example usage
    print("IPD-QMA Validation Module")
    print("=" * 50)

    import numpy as np

    # Generate example data with some issues
    np.random.seed(42)
    control = np.concatenate([
        np.random.normal(0, 1, 95),
        [10, -10, 15],  # Outliers
        [np.nan, np.inf]  # Invalid values
    ])
    treatment = np.random.normal(0.5, 1.2, 100)

    # Validate
    validator = IPDQMAValidator(strict=False)
    result = validator.validate_study(control, treatment, "Example Study")

    print(f"\nValidation Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"Quality Score: {result.score:.1f}/100")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings[:5]:
            print(f"  - {w}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors:
            print(f"  - {e}")

    # Suggestions
    suggestions = validator.suggest_improvements(result)
    if suggestions:
        print(f"\nSuggestions:")
        for s in suggestions:
            print(f"  - {s}")

    # Clean data
    print("\n" + "=" * 50)
    print("Cleaning data...")
    control_clean, treatment_clean = clean_data(control, treatment, remove_outliers=True)
    print(f"Original: n_control={len(control)}, n_treatment={len(treatment)}")
    print(f"Cleaned:  n_control={len(control_clean)}, n_treatment={len(treatment_clean)}")

    # Validate cleaned data
    result_clean = validator.validate_study(control_clean, treatment_clean, "Cleaned Study")
    print(f"\nCleaned data quality score: {result_clean.score:.1f}/100")
