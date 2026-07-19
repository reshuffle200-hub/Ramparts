/**
 * Rhinovirus 3Dpol MD Simulation - Main Application Controller
 *
 * Connects all modules together:
 * - PDB fetching
 * - Physics engine
 * - Binding analysis
 * - Visualization
 *
 * CHANGES vs. the original controller:
 * - The 3D view now plays back the simulation as a recorded trajectory:
 *   each completed physics step is streamed into the visualization engine's
 *   playback buffer (see onPhysicsStep + Start/Pause/Reset handlers).
 * - Hydrophobic contacts are drawn in addition to H-bonds/electrostatics.
 * - All visualization calls use the method names the engine now exposes.
 */

import { PDBFetcher } from './modules/pdb-fetcher.js';
import { VisualizationEngine } from './modules/visualization-engine.js';
import { PhysicsEngine } from './modules/physics-engine.js';
import { InhibitorBuilder } from './modules/inhibitor-builder.js';
import { BindingAnalyzer } from './modules/binding-analyzer.js';

// Application state
let appState = {
  structure: null,
  inhibitor: null,
  physics: null,
  visualization: null,
  bindingAnalyzer: null,
  pdbFetcher: new PDBFetcher(),
  isRunning: false,
  trajCount: 0, // how many recorded frames have been streamed to the viewer
};

// DOM elements
const canvas = document.getElementById('canvas');
const pdbIdInput = document.getElementById('pdbId');
const loadBtn = document.getElementById('loadBtn');
const addInhibitorBtn = document.getElementById('addInhibitorBtn');
const inhibitorType = document.getElementById('inhibitorType');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const viewMode = document.getElementById('viewMode');
const resetCameraBtn = document.getElementById('resetCameraBtn');
const zoomFitBtn = document.getElementById('zoomFitBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const exportJsonBtn = document.getElementById('exportJsonBtn');
const exportCsvBtn = document.getElementById('exportCsvBtn');
const exportPdbBtn = document.getElementById('exportPdbBtn');

const simTime = document.getElementById('simTime');
const stepCount = document.getElementById('stepCount');
const tempValue = document.getElementById('tempValue');
const energyValue = document.getElementById('energyValue');
const loadStatus = document.getElementById('loadStatus');
const bindingResult = document.getElementById('bindingResult');

// ============================================================================
// EVENT LISTENERS
// ============================================================================

loadBtn.addEventListener('click', async () => {
  const pdbId = pdbIdInput.value.trim().toUpperCase();

  if (!pdbId || pdbId.length !== 4) {
    showStatus('Invalid PDB ID (must be 4 characters)', 'error');
    return;
  }

  try {
    loadBtn.disabled = true;
    loadBtn.textContent = '⏳ Loading...';
    showStatus(`Fetching ${pdbId}...`, 'info');

    appState.structure = await appState.pdbFetcher.fetch(pdbId);

    // Initialize visualization
    appState.visualization = new VisualizationEngine(canvas, appState.structure);
    appState.visualization.render();
    appState.visualization.renderStructure(appState.structure);
    appState.visualization.zoomToFit();

    // Initialize physics engine
    appState.physics = new PhysicsEngine(appState.structure, {
      temperature: 300,
      timestep: 0.001,
      recordTrajectory: true,
    });

    // Stream recorded frames into the viewer + update metrics each step.
    appState.trajCount = 0;
    appState.physics.onStepComplete = onPhysicsStep;

    showStatus(`✓ Loaded ${pdbId} (${appState.structure.atoms.length} atoms)`, 'success');
    updateMetrics();
  } catch (error) {
    showStatus(`✗ Error: ${error.message}`, 'error');
    console.error('PDB loading error:', error);
  } finally {
    loadBtn.disabled = false;
    loadBtn.textContent = 'Load PDB';
  }
});

addInhibitorBtn.addEventListener('click', () => {
  if (!appState.structure) {
    showStatus('Load a protein structure first', 'error');
    return;
  }

  try {
    const builder = new InhibitorBuilder();
    const scaffoldMap = {
      nucleoside: 'nucleoside-analog',
      protease: 'protease-inhibitor',
      allosteric: 'allosteric-inhibitor',
      generic: 'generic-small-molecule',
    };

    appState.inhibitor = builder
      .loadPreset(scaffoldMap[inhibitorType.value])
      .build();

    // Visualize inhibitor (orange carbons, CPK heteroatoms).
    if (appState.visualization) {
      appState.visualization.renderLigand(appState.inhibitor, 0xffaa00);
    }

    // Initialize binding analyzer
    appState.bindingAnalyzer = new BindingAnalyzer(appState.structure, appState.inhibitor);

    showStatus(`✓ Added inhibitor (${appState.inhibitor.atoms.length} atoms)`, 'success');
  } catch (error) {
    showStatus(`✗ Error: ${error.message}`, 'error');
    console.error('Inhibitor error:', error);
  }
});

startBtn.addEventListener('click', () => {
  if (!appState.physics) {
    showStatus('Load a structure first', 'error');
    return;
  }

  // Reset the playback buffer to the current pose, then start recording +
  // playing back the trajectory as frames are produced.
  appState.trajCount = 0;
  if (appState.visualization) {
    appState.visualization.loadTrajectory([]); // snapshots current positions as baseline
    appState.visualization.play();
  }

  appState.physics.start();
  appState.isRunning = true;
  startBtn.disabled = true;
  pauseBtn.disabled = false;
  pauseBtn.textContent = '⏸ Pause';
  showStatus('Simulation running...', 'success');
});

pauseBtn.addEventListener('click', () => {
  if (!appState.physics) return;

  appState.physics.pause();
  const paused = appState.physics.isPaused;

  // Keep the playback in sync with the simulation.
  if (appState.visualization) {
    if (paused) appState.visualization.pause();
    else appState.visualization.play();
  }

  pauseBtn.textContent = paused ? '▶ Resume' : '⏸ Pause';
  showStatus(paused ? 'Paused' : 'Resumed', 'info');
});

resetBtn.addEventListener('click', () => {
  if (!appState.physics) return;

  appState.physics.stop();
  appState.physics.reset();
  appState.isRunning = false;
  appState.trajCount = 0;

  // Stop playback and restore the structure to its pre-simulation pose.
  if (appState.visualization) {
    appState.visualization.resetTrajectory();
  }

  startBtn.disabled = false;
  pauseBtn.disabled = true;
  pauseBtn.textContent = '⏸ Pause';
  updateMetrics();
  showStatus('Simulation reset', 'info');
});

viewMode.addEventListener('change', (e) => {
  if (appState.visualization) {
    appState.visualization.setViewMode(e.target.value);
  }
});

resetCameraBtn.addEventListener('click', () => {
  if (appState.visualization) {
    appState.visualization.resetCamera();
  }
});

zoomFitBtn.addEventListener('click', () => {
  if (appState.visualization) {
    appState.visualization.zoomToFit();
  }
});

analyzeBtn.addEventListener('click', () => {
  if (!appState.bindingAnalyzer) {
    showStatus('Add an inhibitor first', 'error');
    return;
  }

  try {
    const results = appState.bindingAnalyzer.analyzeBinding();

    let html = `
      <strong>Binding Analysis</strong><br>
      Score: ${results.bindingScore}/10<br>
      H-Bonds: ${results.metrics.hydrogenBonds.count}<br>
      ΔG: ${results.estimatedDeltaG.toFixed(2)} kcal/mol<br>
      Kd: ${results.estimatedKd.toExponential(1)} M<br>
      <em>${results.prediction}</em>
    `;

    bindingResult.innerHTML = html;
    bindingResult.style.display = 'block';

    // Visualize interactions. These setters are additive on the engine, so
    // drawing electrostatics does not wipe the hydrogen bonds.
    if (appState.visualization) {
      const m = results.metrics;
      appState.visualization.drawHydrogenBonds(m.hydrogenBonds.bonds);
      appState.visualization.drawElectrostaticInteractions(
        m.electrostaticInteractions.interactions
      );
      // Hydrophobic contacts if the analyzer provides them.
      const hydro =
        m.hydrophobicContacts?.contacts ||
        m.hydrophobicInteractions?.interactions ||
        m.hydrophobic?.contacts;
      if (hydro) appState.visualization.drawHydrophobicInteractions(hydro);
    }

    console.log('Binding results:', results);
  } catch (error) {
    showStatus(`✗ Analysis error: ${error.message}`, 'error');
    console.error('Analysis error:', error);
  }
});

exportJsonBtn.addEventListener('click', () => {
  if (!appState.physics) {
    showStatus('Run simulation first', 'error');
    return;
  }

  try {
    const trajectory = appState.physics.exportTrajectory();
    const data = JSON.stringify(trajectory, null, 2);
    downloadFile(data, 'trajectory.json', 'application/json');
    showStatus('✓ Exported JSON', 'success');
  } catch (error) {
    showStatus(`✗ Export error: ${error.message}`, 'error');
  }
});

exportCsvBtn.addEventListener('click', () => {
  if (!appState.bindingAnalyzer) {
    showStatus('Run analysis first', 'error');
    return;
  }

  try {
    const csv = appState.bindingAnalyzer.exportCSV();
    downloadFile(csv, 'binding-results.csv', 'text/csv');
    showStatus('✓ Exported CSV', 'success');
  } catch (error) {
    showStatus(`✗ Export error: ${error.message}`, 'error');
  }
});

exportPdbBtn.addEventListener('click', () => {
  if (!appState.structure) {
    showStatus('Load structure first', 'error');
    return;
  }

  try {
    // Simple PDB export (structure only, no trajectory)
    const pdbLines = [];
    appState.structure.atoms.forEach((atom, idx) => {
      const line = `ATOM  ${String(atom.serial).padStart(5)}  ${String(atom.name).padEnd(4)}${String(atom.resName).padEnd(3)} ${atom.chainId}${String(atom.resSeq).padStart(4)}    ${Number(atom.position.x).toFixed(3).padStart(8)}${Number(atom.position.y).toFixed(3).padStart(8)}${Number(atom.position.z).toFixed(3).padStart(8)}  1.00  0.00           ${atom.element.padEnd(2)}\n`;
      pdbLines.push(line);
    });
    pdbLines.push('END\n');

    downloadFile(pdbLines.join(''), 'structure.pdb', 'text/plain');
    showStatus('✓ Exported PDB', 'success');
  } catch (error) {
    showStatus(`✗ Export error: ${error.message}`, 'error');
  }
});

// ============================================================================
// SIMULATION → VIEWER BRIDGE
// ============================================================================

/**
 * Called once per completed physics step. Updates the metric readouts and
 * streams any newly-recorded trajectory frames into the visualization engine
 * so the view plays back the recorded motion.
 *
 * Robust to two physics-engine conventions:
 *  1. onStepComplete is called with the just-recorded frame (fast path).
 *  2. onStepComplete is called with no args → we diff exportTrajectory().
 *
 * If your PhysicsEngine exposes the current frame differently, this is the one
 * place to adjust.
 */
function onPhysicsStep(frame) {
  updateMetrics();
  const viz = appState.visualization;
  if (!viz) return;

  // Fast path: the callback handed us the frame directly.
  if (frame && (Array.isArray(frame) || frame.positions)) {
    viz.appendTrajectory([frame]);
    return;
  }

  // Fallback: pull the recorded trajectory and append only the new frames.
  try {
    const raw = appState.physics.exportTrajectory();
    const arr = Array.isArray(raw)
      ? raw
      : (raw && (raw.frames || raw.trajectory || raw.steps)) || [];
    if (arr.length > appState.trajCount) {
      viz.appendTrajectory(arr.slice(appState.trajCount));
      appState.trajCount = arr.length;
    }
  } catch (e) {
    /* trajectory not available this step — ignore */
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function updateMetrics() {
  if (!appState.physics) return;

  const stats = appState.physics.getStats();
  simTime.textContent = `${stats.simTime.toFixed(2)} ps`;
  stepCount.textContent = stats.stepCount;
  tempValue.textContent = `${stats.temperature.toFixed(0)} K`;
  energyValue.textContent = `${stats.energy.toFixed(2)} kcal/mol`;
}

function showStatus(message, type = 'info') {
  loadStatus.textContent = message;
  loadStatus.className = `status ${type}`;
  loadStatus.style.display = 'block';

  if (type === 'success' || type === 'error') {
    setTimeout(() => {
      loadStatus.style.display = 'none';
    }, 5000);
  }
}

function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

console.log('✓ Rhinovirus 3Dpol MD Simulation Ready');
console.log('  1. Enter PDB ID and load structure (default 1XR5 = HRV14 3Dpol)');
console.log('  2. Add inhibitor');
console.log('  3. Start simulation (view plays back the recorded trajectory)');
console.log('  4. Analyze binding');
