/**
 * IPD-QMA Integration for DTA Pro
 *
 * This file contains the integration code to add IPD-QMA functionality to DTA Pro.
 * To integrate, add this script to DTA Pro and use the applyIPDQMAIntegration() function.
 *
 * Version: 1.0
 */

// ============================================================================
// IPD-QMA INTEGRATION FOR DTA PRO
// ============================================================================

/**
 * Apply IPD-QMA integration to DTA Pro
 * Call this function after DTA Pro has loaded
 */
function applyIPDQMAIntegration() {
    // Add IPD-QMA tab to navigation
    addIPDQMATab();

    // Add IPD-QMA panel
    addIPDQMAPanel();

    // Initialize IPD-QMA state
    window.IPDQMAState = {
        studies: [],
        config: new IPDQMAConfig({
            quantiles: [0.1, 0.25, 0.5, 0.75, 0.9],
            nBootstrap: 200,
            confidenceLevel: 0.95,
            useRandomEffects: true,
            tau2Estimator: 'dl',
            randomSeed: 42
        }),
        results: null,
        isAnalyzing: false
    };

    console.log('IPD-QMA integration applied successfully');
}

/**
 * Add IPD-QMA tab to navigation
 */
function addIPDQMATab() {
    const tabsNav = document.querySelector('.tabs');
    if (!tabsNav) {
        console.error('Tabs navigation not found');
        return;
    }

    // Create IPD-QMA tab button
    const ipdQmaTab = document.createElement('button');
    ipdQmaTab.className = 'tab';
    ipdQmaTab.setAttribute('data-tab', 'ipdQma');
    ipdQmaTab.innerHTML = '📊 IPD-QMA';
    ipdQmaTab.onclick = function() { switchTab('ipdQma'); };

    // Insert before the "Validation" tab
    const validationTab = tabsNav.querySelector('[data-tab="validation"]');
    if (validationTab) {
        tabsNav.insertBefore(ipdQmaTab, validationTab);
    } else {
        tabsNav.appendChild(ipdQmaTab);
    }
}

/**
 * Add IPD-QMA panel content
 */
function addIPDQMAPanel() {
    const main = document.querySelector('.main');
    if (!main) {
        console.error('Main container not found');
        return;
    }

    // Create IPD-QMA panel
    const panel = document.createElement('section');
    panel.id = 'panel-ipdQma';
    panel.className = 'panel';
    panel.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">📊 IPD-QMA: Individual Participant Data Quantile Meta-Analysis</h2>
                <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
                    <button class="btn btn-secondary btn-sm" onclick="loadIPDQMAExample()">Load Example</button>
                    <button class="btn btn-secondary btn-sm" onclick="clearIPDQMAStudies()">Clear All</button>
                    <button class="btn btn-primary btn-sm" onclick="runIPDQMAAnalysis()">▶ Run Analysis</button>
                </div>
            </div>
            <div class="card-body">
                <div class="alert alert-info">
                    <strong>What is IPD-QMA?</strong><br>
                    IPD-QMA detects heterogeneous treatment effects across patient severity distributions.
                    Unlike traditional meta-analysis that focuses on mean differences, IPD-QMA examines
                    treatment effects across multiple quantiles (Q10, Q25, Q50, Q75, Q90) to identify
                    location-scale shifts in treatment effectiveness.
                </div>

                <div class="grid grid-2">
                    <div>
                        <h3 style="margin-bottom:1rem;">Configuration</h3>
                        <div class="form-group">
                            <label class="form-label">Quantiles to Analyze</label>
                            <select id="ipdQmaQuantiles" class="form-select" onchange="updateIPDQMAConfig()">
                                <option value="standard">Standard (5 quantiles: 10%, 25%, 50%, 75%, 90%)</option>
                                <option value="extended">Extended (9 quantiles)</option>
                                <option value="fine">Fine (19 quantiles)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Bootstrap Samples</label>
                            <input type="number" id="ipdQmaBootstrap" class="form-input"
                                   value="200" min="100" max="5000" step="100" onchange="updateIPDQMAConfig()">
                            <small style="color:var(--text-muted)">More samples = more accurate but slower</small>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Model Type</label>
                            <select id="ipdQmaModel" class="form-select" onchange="updateIPDQMAConfig()">
                                <option value="random">Random-Effects</option>
                                <option value="fixed">Fixed-Effect</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Confidence Level</label>
                            <select id="ipdQmaConfidence" class="form-select" onchange="updateIPDQMAConfig()">
                                <option value="0.90">90%</option>
                                <option value="0.95" selected>95%</option>
                                <option value="0.99">99%</option>
                            </select>
                        </div>
                    </div>

                    <div>
                        <h3 style="margin-bottom:1rem;">Study Data</h3>
                        <div class="alert alert-warning" style="margin-bottom:1rem;">
                            <strong>Note:</strong> IPD-QMA requires individual participant data (IPD) -
                            the raw continuous outcome values for each participant in both control
                            and treatment groups.
                        </div>

                        <div id="ipdQmaStudiesContainer" style="max-height:400px;overflow-y:auto;margin-bottom:1rem;">
                            <div class="placeholder">
                                <div class="placeholder-icon">📊</div>
                                <p>No studies loaded. Click "Load Example" or add studies manually.</p>
                            </div>
                        </div>

                        <button class="btn btn-secondary" onclick="addIPDQMAStudyManual()">
                            + Add Study Manually
                        </button>
                    </div>
                </div>

                <div id="ipdQmaResultsSection" style="display:none;margin-top:2rem;">
                    <h3 style="margin-bottom:1rem;">Analysis Results</h3>

                    <div class="stats-grid" style="margin-bottom:1.5rem;">
                        <div class="stat-card">
                            <div class="stat-value accent" id="ipdQmaNSudies">-</div>
                            <div class="stat-label">Studies</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value accent2" id="ipdQmaModelType">-</div>
                            <div class="stat-label">Model</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="ipdQmaSlopeP">-</div>
                            <div class="stat-label">Slope P-value</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="ipdQmaLnvrP">-</div>
                            <div class="stat-label">lnVR P-value</div>
                        </div>
                    </div>

                    <div class="grid grid-2">
                        <div class="card">
                            <div class="card-header">
                                <h4 class="card-title">Fan Plot: Effects Across Quantiles</h4>
                            </div>
                            <div class="card-body">
                                <div id="ipdQmaFanPlot" class="plot-container-sm"></div>
                            </div>
                        </div>

                        <div class="card">
                            <div class="card-header">
                                <h4 class="card-title">Quantile Profile Table</h4>
                            </div>
                            <div class="card-body">
                                <div id="ipdQmaProfileTable" class="table-wrap"></div>
                            </div>
                        </div>
                    </div>

                    <div class="card" style="margin-top:1.5rem;">
                        <div class="card-header">
                            <h4 class="card-title">Interpretation</h4>
                        </div>
                        <div class="card-body">
                            <div id="ipdQmaInterpretation"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    main.appendChild(panel);
}

/**
 * Load example IPD-QMA data
 */
function loadIPDQMAExample() {
    // Generate example data with heterogeneous effects
    const studies = [];
    const nStudies = 10;

    for (let i = 0; i < nStudies; i++) {
        // Simulate skewed distribution with heterogeneous effects
        const baseScale = 0.8 + Math.random() * 0.4;
        const varianceMultiplier = 2.5 + Math.random() * 1.0;

        // Control group
        const control = [];
        for (let j = 0; j < 100; j++) {
            const u = Math.random();
            control.push(-Math.log(u) * baseScale - 1); // Exponential
        }

        // Treatment group: larger variance + location shift
        const treatment = [];
        for (let j = 0; j < 100; j++) {
            const u = Math.random();
            treatment.push((-Math.log(u) * baseScale - 1) * varianceMultiplier + 0.5);
        }

        // Clean data
        const cleanControl = control.filter(v => v > -2 && v < 5);
        const cleanTreatment = treatment.filter(v => v > -2 && v < 10);

        studies.push({
            name: `Study ${i + 1}`,
            control: cleanControl,
            treatment: cleanTreatment
        });
    }

    window.IPDQMAState.studies = studies;
    renderIPDQMAStudies();

    // Show success message
    showIPDQMASuccess(`Loaded ${nStudies} example studies with heterogeneous treatment effects`);
}

/**
 * Render IPD-QMA studies
 */
function renderIPDQMAStudies() {
    const container = document.getElementById('ipdQmaStudiesContainer');
    if (!container) return;

    if (window.IPDQMAState.studies.length === 0) {
        container.innerHTML = `
            <div class="placeholder">
                <div class="placeholder-icon">📊</div>
                <p>No studies loaded. Click "Load Example" or add studies manually.</p>
            </div>
        `;
        return;
    }

    let html = '';
    window.IPDQMAState.studies.forEach((study, index) => {
        html += `
            <div class="study-row" data-index="${index}">
                <div class="study-header">
                    <strong>${study.name}</strong>
                    <button class="btn-remove" onclick="removeIPDQMAStudy(${index})">×</button>
                </div>
                <div class="study-inputs">
                    <div class="study-input-group">
                        <label>Control (n=${study.control.length})</label>
                        <div class="form-input-sm form-input-mono">
                            Mean: ${IPDQMAMath.mean(study.control).toFixed(2)}, SD: ${IPDQMAMath.std(study.control).toFixed(2)}
                        </div>
                    </div>
                    <div class="study-input-group">
                        <label>Treatment (n=${study.treatment.length})</label>
                        <div class="form-input-sm form-input-mono">
                            Mean: ${IPDQMAMath.mean(study.treatment).toFixed(2)}, SD: ${IPDQMAMath.std(study.treatment).toFixed(2)}
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

/**
 * Run IPD-QMA analysis
 */
function runIPDQMAAnalysis() {
    if (window.IPDQMAState.studies.length === 0) {
        showIPDQMAError('Please load or add studies first');
        return;
    }

    window.IPDQMAState.isAnalyzing = true;
    showIPDQMALoading('Running IPD-QMA analysis...');

    // Use setTimeout to allow UI to update
    setTimeout(() => {
        try {
            // Prepare studies data
            const studiesData = window.IPDQMAState.studies.map(s => [s.control, s.treatment]);

            // Create analyzer and run analysis
            const analyzer = new IPDQMA(window.IPDQMAState.config);
            const results = analyzer.fit(studiesData);

            // Store results
            window.IPDQMAState.results = results;
            window.IPDQMAState.analyzer = analyzer;

            // Display results
            displayIPDQMAResults(results);

            showIPDQMASuccess('IPD-QMA analysis completed successfully!');

        } catch (error) {
            console.error('IPD-QMA analysis error:', error);
            showIPDQMAError('Analysis failed: ' + error.message);
        } finally {
            window.IPDQMAState.isAnalyzing = false;
        }
    }, 100);
}

/**
 * Display IPD-QMA results
 */
function displayIPDQMAResults(results) {
    // Show results section
    document.getElementById('ipdQmaResultsSection').style.display = 'block';

    // Update stats
    document.getElementById('ipdQmaNSudies').textContent = results.nStudies;
    document.getElementById('ipdQmaModelType').textContent = results.modelType.replace('_', ' ');
    document.getElementById('ipdQmaSlopeP').textContent = results.slopeTest.p.toFixed(4);
    document.getElementById('ipdQmaLnvrP').textContent = results.lnvrTest.p.toFixed(4);

    // Color code p-values
    const slopeP = results.slopeTest.p;
    document.getElementById('ipdQmaSlopeP').className = 'stat-value ' +
        (slopeP < 0.05 ? 'success' : 'warning');

    const lnvrP = results.lnvrTest.p;
    document.getElementById('ipdQmaLnvrP').className = 'stat-value ' +
        (lnvrP < 0.05 ? 'success' : 'warning');

    // Create fan plot
    createIPDQMAFanPlot(results);

    // Create profile table
    createIPDQMAProfileTable(results);

    // Create interpretation
    createIPDQMAInterpretation(results);
}

/**
 * Create IPD-QMA fan plot
 */
function createIPDQMAFanPlot(results) {
    const container = document.getElementById('ipdQmaFanPlot');
    if (!container) return;

    const profile = results.profile;

    const quantiles = profile.map(p => p.Quantile);
    const effects = profile.map(p => p.Effect);
    const ciLower = profile.map(p => p.CI_Lower);
    const ciUpper = profile.map(p => p.CI_Upper);

    // Create Plotly figure
    const traceEffect = {
        x: quantiles,
        y: effects,
        mode: 'lines+markers',
        name: 'Pooled Effect',
        line: { color: '#00d4aa', width: 3 },
        marker: { size: 10 }
    };

    const traceCI = {
        x: quantiles.concat(quantiles.slice().reverse()),
        y: ciUpper.concat(ciLower.slice().reverse()),
        fill: 'toself',
        type: 'scatter',
        mode: 'none',
        name: '95% CI',
        fillcolor: 'rgba(0, 212, 170, 0.2)',
        line: { color: 'transparent' }
    };

    const layout = {
        title: 'IPD-QMA Fan Plot: Treatment Effects Across Quantiles',
        xaxis: {
            title: 'Quantile',
            tickformat: '.0%'
        },
        yaxis: { title: 'Effect Size' },
        hovermode: 'closest',
        plot_bgcolor: 'var(--bg-primary)',
        paper_bgcolor: 'var(--bg-card)',
        font: { color: 'var(--text-primary)' }
    };

    Plotly.newPlot(container, [traceCI, traceEffect], layout, { responsive: true });
}

/**
 * Create IPD-QMA profile table
 */
function createIPDQMAProfileTable(results) {
    const container = document.getElementById('ipdQmaProfileTable');
    if (!container) return;

    const profile = results.profile;

    let html = '<table><thead><tr>';
    html += '<th>Quantile</th><th>Effect</th><th>SE</th><th>95% CI</th><th>P-value</th><th>I²</th>';
    html += '</tr></thead><tbody>';

    profile.forEach(p => {
        html += '<tr>';
        html += `<td>${(p.Quantile * 100).toFixed(0)}%</td>`;
        html += `<td class="mono">${p.Effect.toFixed(3)}</td>`;
        html += `<td class="mono">${p.SE.toFixed(3)}</td>`;
        html += `<td class="mono">[${p.CI_Lower.toFixed(3)}, ${p.CI_Upper.toFixed(3)}]</td>`;
        html += `<td class="mono">${p.P.toFixed(4)}</td>`;
        html += `<td class="mono">${p.I2.toFixed(1)}%</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

/**
 * Create IPD-QMA interpretation
 */
function createIPDQMAInterpretation(results) {
    const container = document.getElementById('ipdQmaInterpretation');
    if (!container) return;

    let html = '<div class="grid grid-2">';

    // Slope test
    html += '<div class="stat-card" style="text-align:left;padding:1.5rem;">';
    html += '<h4 style="margin-bottom:0.5rem;">Slope Test (Heterogeneity)</h4>';
    html += `<p style="font-size:0.9rem;">${results.slopeTest.interpretation}</p>`;
    html += `<p style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;">`;
    html += `Estimate: ${results.slopeTest.estimate.toFixed(4)}, P-value: ${results.slopeTest.p.toFixed(4)}</p>`;
    html += '</div>';

    // lnVR test
    html += '<div class="stat-card" style="text-align:left;padding:1.5rem;">';
    html += '<h4 style="margin-bottom:0.5rem;">Log Variance Ratio (Scale Shift)</h4>';
    html += `<p style="font-size:0.9rem;">${results.lnvrTest.interpretation}</p>`;
    html += `<p style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;">`;
    html += `Estimate: ${results.lnvrTest.estimate.toFixed(4)}, P-value: ${results.lnvrTest.p.toFixed(4)}</p>`;
    html += '</div>';

    html += '</div>';

    // Overall interpretation
    html += '<div class="alert ' + (
        (results.slopeTest.p < 0.05 || results.lnvrTest.p < 0.05) ? 'alert-success' : 'alert-info'
    ) + '" style="margin-top:1rem;">';
    html += '<strong>Summary:</strong> ';

    if (results.slopeTest.p < 0.05 && results.lnvrTest.p < 0.05) {
        html += 'Strong evidence of heterogeneous treatment effects - both location and scale shifts detected.';
    } else if (results.slopeTest.p < 0.05) {
        html += 'Treatment effects vary significantly across patient severity quantiles.';
    } else if (results.lnvrTest.p < 0.05) {
        html += 'Significant variance difference between treatment and control groups.';
    } else {
        html += 'No significant heterogeneity detected - treatment effects appear constant across severity levels.';
    }

    html += '</div>';

    container.innerHTML = html;
}

/**
 * Update IPD-QMA configuration
 */
function updateIPDQMAConfig() {
    const quantilesSelect = document.getElementById('ipdQmaQuantiles');
    const bootstrapInput = document.getElementById('ipdQmaBootstrap');
    const modelSelect = document.getElementById('ipdQmaModel');
    const confidenceSelect = document.getElementById('ipdQmaConfidence');

    // Set quantiles
    let quantiles;
    switch (quantilesSelect.value) {
        case 'extended':
            quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
            break;
        case 'fine':
            quantiles = Array.from({length: 19}, (_, i) => 0.05 + i * 0.05);
            break;
        default:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9];
    }

    window.IPDQMAState.config = new IPDQMAConfig({
        quantiles: quantiles,
        nBootstrap: parseInt(bootstrapInput.value),
        useRandomEffects: modelSelect.value === 'random',
        confidenceLevel: parseFloat(confidenceSelect.value),
        tau2Estimator: 'dl',
        randomSeed: 42
    });
}

/**
 * Remove IPD-QMA study
 */
function removeIPDQMAStudy(index) {
    window.IPDQMAState.studies.splice(index, 1);
    renderIPDQMAStudies();
}

/**
 * Clear all IPD-QMA studies
 */
function clearIPDQMAStudies() {
    window.IPDQMAState.studies = [];
    window.IPDQMAState.results = null;
    renderIPDQMAStudies();
    document.getElementById('ipdQmaResultsSection').style.display = 'none';
    showIPDQMASuccess('All studies cleared');
}

/**
 * Add IPD-QMA study manually (simplified - would need proper UI)
 */
function addIPDQMAStudyManual() {
    showIPDQMAError('Manual study entry requires importing IPD data. Please use "Load Example" for demonstration.');
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showIPDQMALoading(message) {
    // Use DTA Pro's existing notification system if available
    if (typeof showNotification === 'function') {
        showNotification(message, 'info');
    } else {
        console.log(message);
    }
}

function showIPDQMASuccess(message) {
    if (typeof showNotification === 'function') {
        showNotification(message, 'success');
    } else {
        alert(message);
    }
}

function showIPDQMAError(message) {
    if (typeof showNotification === 'function') {
        showNotification(message, 'error');
    } else {
        alert('Error: ' + message);
    }
}

// ============================================================================
// AUTO-APPLY INTEGRATION
// ============================================================================

// Auto-apply when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyIPDQMAIntegration);
} else {
    applyIPDQMAIntegration();
}

// Export for external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { applyIPDQMAIntegration, addIPDQMATab, addIPDQMAPanel };
}
