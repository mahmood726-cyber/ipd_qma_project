"""IPD-QMA machine-learning helpers.

Reconstructed header (2026-04-14) — the original file had been truncated
from line 1 and started mid-expression with `.asarray(treatment)`. This
header makes the remainder of the file parseable again. Semantics of
assess_study_quality are inferred from the existing method body
(iterates (control, treatment) pairs, extracts features, appends
quality_scores dicts, classifies by threshold).
"""
from typing import List, Tuple

import numpy as np
from scipy import stats
from scipy.stats import kurtosis, skew


class IPDQMAQualityAssessor:
    """Feature-based quality assessment for IPD meta-analysis studies."""

    def assess_study_quality(
        self,
        studies_data: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[dict]:
        """Score each (control, treatment) study on sample size, balance,
        distribution shape, and outlier presence. Returns a list of dicts
        with a `quality_score` in [0, 120] and `is_high_quality` /
        `is_suspicious` flags."""
        quality_scores: List[dict] = []
        for i, (control, t) in enumerate(studies_data):
            control = np.asarray(control)
            treatment = np.asarray(t)

            # Feature extraction for quality assessment
            features = [
                len(control) / 100,
                # Sample size
                len(treatment) / 100,
                # Effect size
                abs(np.mean(treatment) - np.mean(control)) / np.sqrt(np.var(control)),
                # Variance ratio
                np.var(treatment) / np.var(control),
                # Skewness of difference
                stats.skew(treatment - control),
                # Kurtosis of difference
                stats.kurtosis(treatment - control),
                # Outliers
                self._detect_outliers_count(treatment - control),
                # Normality test p-value
                stats.shapiro(treatment - control)[1]
            ]

            quality_scores.append({
                'study_id': i,
                'control_n': len(control),
                'treatment_n': len(t),
                'quality_score': sum([
                    max(0, min(20, len(control) - 100)),  # Sample size score (max 20)
                    max(0, min(20, len(treatment) - 100)),  # Sample size score
                    max(0, min(20, len(control) + len(treatment) - 200)),  # Total sample size score
                    20 - abs(skew(treatment - control)),  # Symmetry score
                    20 - abs(kurtosis(treatment - control)),  # Normality score
                    20 - self._detect_outliers_count(treatment - control),  # No outliers
                ]),
                'features': features,
                'is_high_quality': None
            })

        # Classify using thresholds
        for score in quality_scores:
            score['is_high_quality'] = score['quality_score'] >= 80
            score['is_suspicious'] = score['quality_score'] <= 50

        return quality_scores

    def _detect_outliers_count(self, data: np.ndarray) -> int:
        """Count outliers using multiple methods."""
        if len(data) == 0:
            return 0

        # Method 1: IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr_count = np.sum((data < q1 - 1.5 * (q3 - q1)) |
                         (data > q3 + 1.5 * (q3 - q1)))

        # Method 2: Z-score method
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        z_count = np.sum(z_scores > 3)

        # Return count
        return max(iqr_count, z_count)

    def identify_suspicious_studies(
        self,
        studies_data: List[Tuple[np.ndarray, np.ndarray]],
        threshold: int = 3
    ) -> List[int]:
        """
        Identify potentially suspicious studies using anomaly detection.

        Uses Isolation Forest for anomaly detection in high-dimensional space.

        Parameters
        **Parameters**
        studies_data : list of tuples
            List of (control, treatment) pairs
        **threshold** : int
            Number of standard deviations for anomaly

        Returns
        **list**
            Indices of suspicious studies
        """
        if len(studies_data) < 3:
            return []

        # Extract features for each study
        study_features = []

        for i, (control, treatment) in enumerate(studies_data):
            control = np.asarray(control)
            treatment = np.ndarray

            # Calculate features
            features = []
            features.extend([
                np.mean(treatment) - np.mean(control),
                np.std(treatment) / np.std(control) - 1,
                len(treatment) / len(control),
                np.sqrt(np.var(treatment) / np.var(control)),
                stats.skew(treatment - control),
                stats.kurtosis(treatment - control),
                np.percentile(control, 10),
                np.percentile(control, 90),
                np.percentile(treatment, 10),
                np.percentile(treatment, 90)
            ])

            # Extract summary statistics across quantiles
            config = IQMAConfig(
                quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                n_bootstrap=100,
                use_random_effects=False,
                show_progress=False
            )

            try:
                analyzer = IPDQMA(config)
                result = analyzer.analyze_study(control, treatment)
                features.extend([
                    result['slope'],
                    result['lnvr'],
                    result['slope'] / (result['se_slope'] if result['se_slope'] > 0 else 1)
                ])
            except:
                pass

            # Add to list
            study_features.append(features)

        # Convert to numpy array
        X = np.array(study_features)

        # Scale features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

        # Handle edge cases
        if X.shape[1] < 3 or X.shape[0] < 1:
            return []

        # Scale features
        X_scaled = scaler.fit_transform(X)

        # Fit Isolation Forest
        iso = IsolationForest(
            n_estimators=100,
            max_samples=256,
            contamination=0.1,
            n_jobs=-1,
            random_state=42,
            behaviour='new'
        )

        # Fit model
        iso.fit(X_scaled)

        # Identify anomalies
        anomalies = iso.predict(X_scaled)

        # Get outlier scores
        iso_scores = iso.score_samples(X_scaled)
        anomaly_indices = np.where(iso_scores < -threshold)[0]

        return sorted(anomaly_indices)

    def optimize_quantiles(
        self,
        control: np.ndarray,
       _treatment: np.ndarray,
        n_quantiles: int = 5
    ) -> List[float]:
        """
        Use genetic algorithm to find optimal quantile positions.

        Uses differential evolution to optimize quantile selection
        to maximize detection of heterogeneous effects.

        Parameters
        ----------
        control : array-like
            Control group outcomes
        treatment : array-like
            Treatment group outcomes
        n_quantiles : int
            Number of quantiles (between 3 and 20)

        Returns
        -------
        list
            Optimal quantile positions
        """
        from scipy.optimize import differential_evolution

        def objective(quantiles_list):
            # Ensure sorted quantiles
            quantiles = np.array(quantiles)
            if np.any(quantiles[:-1] >= quantiles[1:]):
                return 1000  # Penalty for non-sorted quantiles

            # Ensure range
            if np.any(quantiles < 0.01) or np.any(quantiles > 0.99):
                return 1000  # Penalty for out-of-range quantiles

            # Run analysis
            config = IQMAConfig(
                quantiles=quantiles.tolist(),
                n_bootstrap=200,
                use_random_effects=True,
                show_progress=False
            )

            analyzer = IPDQMA(config)
            results = analyzer.fit([(control, treatment)])

            # Score this configuration
            # Score: maximize detection of heterogeneous effects
            score = 0

            # Slope test p-value
            if results['slope_test']['p'] < 0.05:
                score += 50

            # lnVR test p-value
            if results['lnvr_test']['p'] < 0.05:
                score += 50

            # Effect size
            effects = [r['Effect'] for r in results['profile']]
            max_effect = max(abs(e) for e in effects)
            if max_effect > 0.3:
                score += 10
            elif max_effect > 0.5:
                score += 20

            return score

        bounds = [(0.01, 0.99)]

        # Optimize quantiles
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            lambda q: objective([q[i] for i in range(len(q))]),
            bounds=bounds,
            seed=42
        )

        optimal_quantiles = np.array(result.x)

        return sorted(optimal_quantiles.tolist())


def run_heterogeneity_detection_demo():
    """
    Demonstration of ML-based heterogeneity detection.
    """
    print("=" * 60)
    print("IPD-QMA ML: Automated Heterogeneity Detection")
    print("=" * 60)
    print()

    # Generate simulated data with heterogeneity patterns
    np.random.seed(42)

    # Homogeneous data
    control_homo = np.random.normal(0, 1, 100)
    treatment_homo = np.random.normal(0.5, 1, 100)

    # Heterogeneous data
    control_hetero = np.random.normal(0, 1, 100)
    treatment_hetero = np.random.normal(0.5, 1.2, 100)

    # Extreme heterogeneity
    control_extreme = np.random.exponential(1, 100) - 1
    treatment_extreme = np.random.exponential(2, 100) - 1

    print("1. Create simulation data:")
    print("   - Homogeneous: mean=0.5, SD=1")
    print("   - Heterogeneous: mean=0.5, SD varies")
    print("   - Extreme heterogeneity: exponential distributions")
    print()

    # Create ML model
    ml_config = MLConfig(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1
    )

    ml_analyzer = IPDQMAAutoAnalyzer(ml_config)

    # Create training data
    simulated_data = []
    for control, treatment in [(control_homo, treatment_homo),
                                  (control_hetero, treatment_hetero),
                                  (control_extreme, treatment_extreme)]:
        simulated_data.append({
            'control': control.copy(),
            'treatment': treatment.copy(),
            'is_heterogeneous': control_hetero is control_homo,
            'profile': None
        })

    # Train classifier
    print("2. Training ML classifier...")
    training_results = ml_analyzer.train_heterogeneity_classifier(simulated_data)

    print(f"   Cross-validation scores: {training_results['mean_cv_score']:.3f}")
    print(f"   Feature importance:")
    for name, score in training_results['feature_importance'].items():
        print(f"     {name}: {score:.3f}")

    # Test on heterogeneous data
    print("\n3. Testing on heterogeneous data...")
    detection = ml_analyzer.detect_heterogeneity_auto(
        control_hetero,
        treatment_hetero
    )

    print(f"   Probability of heterogeneity: {detection['probability']:.3f}")
    print(f"   Confidence: {detection['confidence']:.3f}")
    print(f"   Top features: {detection['top_features']}")

    # Compare with traditional tests
    config = IQMAConfig(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        n_bootstrap=500,
        use_random_effects=True,
    )

    analyzer = IPDQMA(config)
    traditional_results = analyzer.fit([
        (control_hetero, treatment_hetero)
    ])

    print(f"\n4. Traditional IPD-QMA:")
    print(f"   Slope test p-value: {traditional_results['slope_test']['p']:.4f}")
    print(f"   lnVR test p-value: {traditional_results['lnvr_test']['p']:.4f}")

    print(f"\n5. Classification vs Traditional Tests:")
    if detection['is_heterogeneous'] and traditional_results['slope_test']['p'] < 0.05:
        print("   ✓ Both methods detect heterogeneity")
    elif detection['is_heterogeneous'] and traditional_results['slope_test']['p'] >= 0.05:
        print("   ✓ ML detected heterogeneity, traditional did not")
    else:
        print("   ✗ ML and traditional agree: no heterogeneity")

    # Test on homogeneous data
    print(f"\n6. Testing on homogeneous data...")
    detection_homo = ml_analyzer.detect_heterogeneity_auto(
        control_homo,
        treatment_homo
    )

    print(f"   Probability of heterogeneity: {detection_homo['probability']:.3f}")
    print(f"   Confidence: {detection_homo['confidence']:.3f}")
    print(f"   Is heterogeneous: {detection_homo['is_heterogeneous']}")

    if detection_homo['is_heterogeneous']:
        print("   ✗ False positive (ML incorrectly identifies heterogeneity)")
    else:
        print("   ✓ Correctly identifies no heterogeneity")

    print(f"\n7. Testing on extreme heterogeneity...")
    detection_extreme = ml_analyzer.detect_heterogeneity_auto(
        control_extreme,
        treatment_extreme
    )

    print(f"   Probability of heterogeneity: {detection_extreme['probability']:.3f}")
    print(f"   Confidence: {detection_extreme['confidence']:.3f}")
    print(f"   Is heterogeneous: {detection_extreme['is_heterogeneous']}")

    if detection_extreme['is_heterogeneous']:
        print("   ✓ Correctly identifies heterogeneity")
    else:
        print("   ✗ Failed to detect extreme heterogeneity")

    print("\n" + "=" * 60)
    print("ML heterogeneity detection demo complete!")
    print("=" * 60)


# Run demonstration
if __name__ == "__main__":
    run_heterogeneity_detection_demo()
