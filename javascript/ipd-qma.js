/**
 * IPD-QMA JavaScript Implementation
 *
 * A JavaScript port of the IPD-QMA (Individual Participant Data Quantile Meta-Analysis)
 * library for detecting heterogeneous treatment effects across patient severity
 * distributions using quantile-based analysis with bootstrap inference.
 *
 * Version: 2.0
 *
 * @module IPDQMA
 */

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const IPDQMAMath = {
    /**
     * Calculate mean of array
     */
    mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    },

    /**
     * Calculate standard deviation
     */
    std(arr, ddof = 1) {
        const m = this.mean(arr);
        const variance = arr.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / (arr.length - ddof);
        return Math.sqrt(variance);
    },

    /**
     * Calculate percentile
     */
    percentile(arr, p) {
        const sorted = [...arr].sort((a, b) => a - b);
        const idx = p * (sorted.length - 1);
        const lower = Math.floor(idx);
        const upper = Math.ceil(idx);
        const weight = idx - lower;

        if (upper >= sorted.length) {
            return sorted[sorted.length - 1];
        }

        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    },

    /**
     * Calculate multiple percentiles
     */
    percentiles(arr, quantiles) {
        return quantiles.map(q => this.percentile(arr, q));
    },

    /**
     * Generate random integer
     */
    randomInt(min, max) {
        return Math.floor(Math.random() * (max - min)) + min;
    },

    /**
     * Normal distribution random (Box-Muller transform)
     */
    randomNormal(mean = 0, std = 1) {
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return z0 * std + mean;
    },

    /**
     * Standard normal CDF approximation
     */
    normalCDF(x) {
        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;
        const p = 0.3275911;

        const sign = x < 0 ? -1 : 1;
        x = Math.abs(x) / Math.sqrt(2);

        const t = 1 / (1 + p * x);
        const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return 0.5 * (1 + sign * y);
    },

    /**
     * Inverse normal CDF (approximation)
     */
    normalInverseCDF(p) {
        if (p <= 0) return -Infinity;
        if (p >= 1) return Infinity;

        if (p === 0.5) return 0;

        // Approximation from Abramowitz and Stegun
        const a = [-3.969683028665376e+01, 2.209460984245205e+02,
                   -2.759285104469687e+02, 1.383577518672690e+02,
                   -3.066479806614716e+01, 2.506628277459239e+00];
        const b = [-5.447609879822406e+01, 1.615858368580409e+02,
                   -1.556989798598866e+02, 6.680131188771972e+01,
                   -1.328068155288572e+01];
        const c = [-7.784894002430293e-03, -3.223964580411365e-01,
                   -2.400758277161838e+00, -2.549732539343734e+00,
                   4.374664141464968e+00, 2.938163982698783e+00];
        const d = [7.784695709041462e-03, 3.224671290700398e-01,
                   2.445134137142996e+00, 3.754408661907416e+00];

        const pLow = 0.02425;
        const pHigh = 1 - pLow;
        let q, r;

        if (p < pLow) {
            q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        } else if (p <= pHigh) {
            q = p - 0.5;
            r = q * q;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        } else {
            q = Math.sqrt(-2 * Math.log(1 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
    },

    /**
     * Chi-squared CDF
     */
    chi2CDF(x, df) {
        // Approximation for chi-squared distribution
        if (x <= 0) return 0;
        if (df === 1) {
            const z = Math.sqrt(x);
            return 2 * (this.normalCDF(z) - 0.5);
        }
        // Wilson-Hilferty approximation
        const z = (Math.pow(x / df, 1/3) - (1 - 2/(9*df))) / Math.sqrt(2/(9*df));
        return this.normalCDF(z);
    }
};

// ============================================================================
// IPD-QMA CONFIG CLASS
// ============================================================================

class IPDQMAConfig {
    constructor(options = {}) {
        this.quantiles = options.quantiles || [0.1, 0.25, 0.5, 0.75, 0.9];
        this.nBootstrap = options.nBootstrap || 200;
        this.confidenceLevel = options.confidenceLevel || 0.95;
        this.randomSeed = options.randomSeed || null;
        this.useRandomEffects = options.useRandomEffects !== undefined ? options.useRandomEffects : true;
        this.tau2Estimator = options.tau2Estimator || 'dl';
    }

    get zAlpha() {
        return IPDQMAMath.normalInverseCDF(1 - (1 - this.confidenceLevel) / 2);
    }
}

// ============================================================================
// MAIN IPD-QMA CLASS
// ============================================================================

class IPDQMA {
    constructor(config) {
        this.config = config || new IPDQMAConfig();
        this.results = null;
        this.studyResults = [];

        if (this.config.randomSeed !== null) {
            this.seed(this.config.randomSeed);
        }
    }

    /**
     * Seed random number generator (simple implementation)
     */
    seed(seed) {
        // Simple seeding - for better results, use a proper RNG library
        this._seed = seed;
        this._random = () => {
            this._seed = (this._seed * 9301 + 49297) % 233280;
            return this._seed / 233280;
        };
        // Override Math.random temporarily
        this._originalRandom = Math.random;
        Math.random = this._random;
    }

    /**
     * Restore original Math.random
     */
    _restoreRandom() {
        if (this._originalRandom) {
            Math.random = this._originalRandom;
        }
    }

    /**
     * Validate input data
     */
    _validateInputs(control, treatment) {
        if (control.length < 10) {
            console.warn(`Small control sample size (n=${control.length})`);
        }
        if (treatment.length < 10) {
            console.warn(`Small treatment sample size (n=${treatment.length})`);
        }

        // Check for NaN/Infinity
        const hasNaN = control.some(isNaN) || treatment.some(isNaN);
        const hasInf = control.some(isFinite) === false || treatment.some(isFinite) === false;

        if (hasNaN || hasInf) {
            throw new Error("Data contains NaN or infinite values");
        }
    }

    /**
     * Analyze a single study
     */
    analyzeStudy(control, treatment) {
        this._validateInputs(control, treatment);

        const nC = control.length;
        const nT = treatment.length;

        // Bootstrap quantile estimation
        const bootQC = [];
        const bootQT = [];
        const nBoot = this.config.nBootstrap;

        for (let i = 0; i < nBoot; i++) {
            // Bootstrap resampling
            const bootC = [];
            const bootT = [];
            for (let j = 0; j < nC; j++) {
                bootC.push(control[Math.floor(Math.random() * nC)]);
            }
            for (let j = 0; j < nT; j++) {
                bootT.push(treatment[Math.floor(Math.random() * nT)]);
            }

            bootQC.push(IPDQMAMath.percentiles(bootC, this.config.quantiles));
            bootQT.push(IPDQMAMath.percentiles(bootT, this.config.quantiles));
        }

        // Calculate differences (effects)
        const bootDiffs = bootQC.map((qc, i) =>
            qc.map((val, j) => bootQT[i][j] - val)
        );

        // Point estimates
        const obsQC = IPDQMAMath.percentiles(control, this.config.quantiles);
        const obsQT = IPDQMAMath.percentiles(treatment, this.config.quantiles);
        const obsEffects = obsQC.map((val, i) => obsQT[i] - val);

        // Standard errors
        const seEffects = bootDiffs[0].map((_, i) =>
            IPDQMAMath.std(bootDiffs.map(row => row[i]))
        );

        // Slope (Q90 - Q10)
        const bootSlopes = bootDiffs.map(row => row[row.length - 1] - row[0]);
        const obsSlope = obsEffects[obsEffects.length - 1] - obsEffects[0];
        const seSlope = IPDQMAMath.std(bootSlopes);

        // Log variance ratio
        const varT = IPDQMAMath.std(treatment) ** 2;
        const varC = IPDQMAMath.std(control) ** 2;
        const lnVR = varC > 0 && varT > 0 ? Math.log(varT / varC) : 0;
        const seLnVR = Math.sqrt(1 / (nT - 1) + 1 / (nC - 1));

        // Bias correction
        const biasCorrection = this._calculateBiasCorrection(bootDiffs, obsEffects);

        return {
            quantiles: obsEffects,
            seQuantiles: seEffects,
            quantileEffectsBC: obsEffects.map((e, i) => e - biasCorrection[i]),
            slope: obsSlope,
            seSlope: seSlope,
            lnVR: lnVR,
            seLnVR: seLnVR,
            nControl: nC,
            nTreatment: nT,
            meanControl: IPDQMAMath.mean(control),
            meanTreatment: IPDQMAMath.mean(treatment),
            sdControl: IPDQMAMath.std(control),
            sdTreatment: IPDQMAMath.std(treatment)
        };
    }

    /**
     * Calculate bias correction
     */
    _calculateBiasCorrection(bootDiffs, obsEffects) {
        return obsEffects.map((obs, i) => {
            const propLess = bootDiffs.reduce((sum, row) =>
                sum + (row[i] < obs ? 1 : 0), 0) / bootDiffs.length;
            const pPropLess = Math.max(0.001, Math.min(0.999, propLess));
            const z0 = IPDQMAMath.normalInverseCDF(pPropLess);
            return -z0 * IPDQMAMath.std(bootDiffs.map(row => row[i]));
        });
    }

    /**
     * Fixed-effect pooling
     */
    _poolFixedEffect(estimates, se) {
        const weights = estimates.map((_, i) => 1 / (se[i] * se[i]));
        const sumWeights = weights.reduce((a, b) => a + b, 0);

        const pooled = estimates.reduce((sum, est, i) => sum + est * weights[i], 0) / sumWeights;
        const sePooled = Math.sqrt(1 / sumWeights);
        const z = pooled / sePooled;
        const pValue = 2 * (1 - IPDQMAMath.normalCDF(Math.abs(z)));

        const ciMargin = this.config.zAlpha * sePooled;

        return {
            estimate: pooled,
            se: sePooled,
            z: z,
            p: pValue,
            lower: pooled - ciMargin,
            upper: pooled + ciMargin,
            weights: weights.map(w => w / sumWeights)
        };
    }

    /**
     * Estimate heterogeneity (tau2)
     */
    _estimateHeterogeneity(estimates, se) {
        const k = estimates.length;
        if (k < 2) {
            return { tau2: 0, tau: 0, i2: 0, q: 0, qP: 1 };
        }

        // Q statistic
        const weights = estimates.map((_, i) => 1 / (se[i] * se[i]));
        const sumWeights = weights.reduce((a, b) => a + b, 0);
        const pooledFixed = estimates.reduce((sum, est, i) => sum + est * weights[i], 0) / sumWeights;

        const q = estimates.reduce((sum, est, i) =>
            sum + weights[i] * Math.pow(est - pooledFixed, 2), 0);
        const qP = 1 - IPDQMAMath.chi2CDF(q, k - 1);

        // DerSimonian-Laird estimator
        const c = sumWeights - weights.reduce((sum, w) => sum + w * w, 0) / sumWeights;
        const tau2 = Math.max(0, (q - (k - 1)) / c);

        // I²
        const i2 = q > (k - 1) ? Math.max(0, (q - (k - 1)) / q * 100) : 0;

        return {
            tau2: tau2,
            tau: Math.sqrt(tau2),
            i2: i2,
            q: q,
            qP: qP
        };
    }

    /**
     * Random-effects pooling
     */
    _poolRandomEffects(estimates, se, het) {
        const tau2 = het.tau2;
        const weightsRE = estimates.map((_, i) => 1 / (se[i] * se[i] + tau2));
        const sumWeights = weightsRE.reduce((a, b) => a + b, 0);

        const pooled = estimates.reduce((sum, est, i) => sum + est * weightsRE[i], 0) / sumWeights;
        const sePooled = Math.sqrt(1 / sumWeights);
        const z = pooled / sePooled;
        const pValue = 2 * (1 - IPDQMAMath.normalCDF(Math.abs(z)));

        const ciMargin = this.config.zAlpha * sePooled;

        // Prediction interval
        const sePred = Math.sqrt(sePooled * sePooled + tau2);
        const predMargin = this.config.zAlpha * sePred;

        return {
            estimate: pooled,
            se: sePooled,
            z: z,
            p: pValue,
            lower: pooled - ciMargin,
            upper: pooled + ciMargin,
            predLower: pooled - predMargin,
            predUpper: pooled + predMargin,
            weights: weightsRE.map(w => w / sumWeights)
        };
    }

    /**
     * Fit IPD-QMA model to multiple studies
     */
    fit(studiesData) {
        // Analyze each study
        this.studyResults = studiesData.map(study =>
            this.analyzeStudy(study[0], study[1])
        );

        const nStudies = this.studyResults.length;

        // Pool quantiles
        const qSummary = [];
        for (let i = 0; i < this.config.quantiles.length; i++) {
            const est = this.studyResults.map(s => s.quantiles[i]);
            const se = this.studyResults.map(s => s.seQuantiles[i]);

            let result, het;
            if (this.config.useRandomEffects) {
                het = this._estimateHeterogeneity(est, se);
                result = this._poolRandomEffects(est, se, het);
            } else {
                result = this._poolFixedEffect(est, se);
                het = this._estimateHeterogeneity(est, se);
            }

            qSummary.push({
                Quantile: this.config.quantiles[i],
                Effect: result.estimate,
                SE: result.se,
                Z: result.z,
                P: result.p,
                CI_Lower: result.lower,
                CI_Upper: result.upper,
                Pred_Lower: result.predLower || null,
                Pred_Upper: result.predUpper || null,
                I2: het.i2,
                Tau2: het.tau2,
                Q: het.q,
                Q_P: het.qP
            });
        }

        // Pool slope
        const sEst = this.studyResults.map(s => s.slope);
        const sSE = this.studyResults.map(s => s.seSlope);
        const sHet = this._estimateHeterogeneity(sEst, sSE);
        const slopeResult = this.config.useRandomEffects ?
            this._poolRandomEffects(sEst, sSE, sHet) :
            this._poolFixedEffect(sEst, sSE);

        // Pool lnVR
        const lEst = this.studyResults.map(s => s.lnVR);
        const lSE = this.studyResults.map(s => s.seLnVR);
        const lHet = this._estimateHeterogeneity(lEst, lSE);
        const lnvrResult = this.config.useRandomEffects ?
            this._poolRandomEffects(lEst, lSE, lHet) :
            this._poolFixedEffect(lEst, lSE);

        this.results = {
            nStudies: nStudies,
            modelType: this.config.useRandomEffects ? 'random_effects' : 'fixed_effect',
            profile: qSummary,
            slopeTest: {
                estimate: slopeResult.estimate,
                se: slopeResult.se,
                p: slopeResult.p,
                ciLower: slopeResult.lower,
                ciUpper: slopeResult.upper,
                i2: sHet.i2,
                tau2: sHet.tau2,
                qP: sHet.qP,
                interpretation: this._interpretSlope(slopeResult.estimate, slopeResult.p)
            },
            lnvrTest: {
                estimate: lnvrResult.estimate,
                se: lnvrResult.se,
                p: lnvrResult.p,
                ciLower: lnvrResult.lower,
                ciUpper: lnvrResult.upper,
                i2: lHet.i2,
                tau2: lHet.tau2,
                qP: lHet.qP,
                interpretation: this._interpretLnVR(lnvrResult.estimate, lnvrResult.p)
            },
            studyDetails: this.studyResults,
            config: this.config
        };

        this._restoreRandom();
        return this.results;
    }

    /**
     * Interpret slope test result
     */
    _interpretSlope(estimate, pValue) {
        if (pValue < 0.001) {
            return `Very strong evidence of heterogeneous effects (slope=${estimate.toFixed(3)}, p<0.001)`;
        } else if (pValue < 0.05) {
            return `Significant heterogeneous effects detected (slope=${estimate.toFixed(3)}, p=${pValue.toFixed(4)})`;
        } else if (pValue < 0.10) {
            return `Trend toward heterogeneous effects (slope=${estimate.toFixed(3)}, p=${pValue.toFixed(4)})`;
        } else {
            return `No significant heterogeneity detected (slope=${estimate.toFixed(3)}, p=${pValue.toFixed(4)})`;
        }
    }

    /**
     * Interpret lnVR test result
     */
    _interpretLnVR(estimate, pValue) {
        const direction = estimate > 0 ? "increased" : "decreased";
        if (pValue < 0.001) {
            return `Very strong evidence of variance difference (lnVR=${estimate.toFixed(3)}, p<0.001)`;
        } else if (pValue < 0.05) {
            return `Significant variance ${direction} (lnVR=${estimate.toFixed(3)}, p=${pValue.toFixed(4)})`;
        } else if (pValue < 0.10) {
            return `Trend toward variance difference (lnVR=${estimate.toFixed(3)}, p=${pValue.toFixed(4)})`;
        } else {
            return `No significant variance difference (lnVR=${estimate.toFixed(3)}, p=${pValue.toFixed(4)})`;
        }
    }

    /**
     * Get summary table
     */
    summary() {
        if (!this.results) {
            throw new Error("Run fit() first.");
        }

        let output = `\n${'='.repeat(60)}\n`;
        output += `IPD-QMA Analysis Summary (${this.results.nStudies} studies)\n`;
        output += `${'='.repeat(60)}\n`;
        output += `\nSlope Test (Heterogeneity):\n`;
        output += `  Estimate: ${this.results.slopeTest.estimate.toFixed(4)}\n`;
        output += `  P-value: ${this.results.slopeTest.p.toFixed(4)}\n`;
        output += `  I²: ${this.results.slopeTest.i2.toFixed(1)}%\n`;
        output += `  ${this.results.slopeTest.interpretation}\n`;
        output += `\nLog Variance Ratio Test (Scale Shift):\n`;
        output += `  Estimate: ${this.results.lnvrTest.estimate.toFixed(4)}\n`;
        output += `  P-value: ${this.results.lnvrTest.p.toFixed(4)}\n`;
        output += `  I²: ${this.results.lnvrTest.i2.toFixed(1)}%\n`;
        output += `  ${this.results.lnvrTest.interpretation}\n`;
        output += `${'='.repeat(60)}\n`;

        return output;
    }
}

// ============================================================================
// EXPORT FOR BROWSER AND NODE
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    // Node.js
    module.exports = { IPDQMA, IPDQMAConfig, IPDQMAMath };
} else {
    // Browser
    window.IPDQMA = IPDQMA;
    window.IPDQMAConfig = IPDQMAConfig;
    window.IPDQMAMath = IPDQMAMath;
}
