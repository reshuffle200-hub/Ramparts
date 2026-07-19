/**
 * visualization-engine.js
 * ---------------------------------------------------------------------------
 * A self-contained WebGL rendering engine for molecular dynamics data,
 * built on Three.js. It renders atoms as instanced spheres and bonds as
 * instanced cylinders, supports three representations (ball-and-stick,
 * space-fill, cartoon ribbon), a trackball-style interactive camera,
 * colour-coded non-covalent interaction overlays, and playback of MD
 * trajectories.
 *
 * Public API (see method JSDoc for details):
 *   const engine = new VisualizationEngine(containerEl, options);
 *   engine.loadStructure(structure);        // receptor / protein
 *   engine.loadInhibitor(inhibitor);        // ligand / small molecule
 *   engine.setInteractions(interactions);   // H-bond / electrostatic / hydrophobic
 *   engine.loadTrajectory(trajectory);      // array of frames
 *   engine.setViewMode('ball-and-stick' | 'space-fill' | 'cartoon');
 *   engine.play(); engine.pause(); engine.setFrame(i);
 *   engine.resetView(); engine.dispose();
 *
 * Expected data shapes:
 *   structure = {
 *     atoms:  [{ id, element, position:{x,y,z}, charge, resName, resSeq, name?, chainId? }, ...],
 *     bonds:  [{ atom1Index, atom2Index, type }, ...],   // indices into atoms[]
 *     residues, chains  // optional metadata
 *   }
 *   inhibitor = { atoms:[...], bonds:[...] }   // same shape as structure
 *   trajectory = [{ step, positions:[{x,y,z}, ...], velocities?:[...] }, ...]
 *   interactions = {
 *     hydrogenBonds: [{ atom1Index, atom2Index }, ...],
 *     electrostatic: [{ atom1Index, atom2Index }, ...],
 *     hydrophobic:   [{ atom1Index, atom2Index }, ...]
 *   }
 *
 * Trajectory frame positions are matched to the *global* atom list, which is
 * the concatenation of the receptor atoms followed by the inhibitor atoms in
 * load order. If a frame's positions length equals the receptor atom count,
 * only the receptor is updated.
 * ---------------------------------------------------------------------------
 */

import * as THREE from 'three';

/* =========================================================================
 * Element data tables (CPK / Jmol-inspired)
 * ========================================================================= */

/** Element display colours (hex). Falls back to DEFAULT_COLOR. */
const ELEMENT_COLORS = {
  H: 0xffffff, HE: 0xd9ffff, LI: 0xcc80ff, BE: 0xc2ff00, B: 0xffb5b5,
  C: 0x909090, N: 0x3050f8, O: 0xff0d0d, F: 0x90e050, NE: 0xb3e3f5,
  NA: 0xab5cf2, MG: 0x8aff00, AL: 0xbfa6a6, SI: 0xf0c8a0, P: 0xff8000,
  S: 0xffff30, CL: 0x1ff01f, AR: 0x80d1e3, K: 0x8f40d4, CA: 0x3dff00,
  FE: 0xe06633, ZN: 0x7d80b0, BR: 0xa62929, I: 0x940094, MN: 0x9c7ac7,
  CU: 0xc88033, CO: 0xf090a0, NI: 0x50d050,
};
const DEFAULT_COLOR = 0xff1493; // hot pink for unknown elements

/** Van der Waals radii in Angstroms (used for space-fill). */
const ELEMENT_VDW = {
  H: 1.20, HE: 1.40, LI: 1.82, C: 1.70, N: 1.55, O: 1.52, F: 1.47,
  NE: 1.54, NA: 2.27, MG: 1.73, SI: 2.10, P: 1.80, S: 1.80, CL: 1.75,
  AR: 1.88, K: 2.75, CA: 2.31, FE: 1.44, ZN: 1.39, BR: 1.85, I: 1.98,
  MN: 1.61, CU: 1.40, NI: 1.63, CO: 1.52,
};
const DEFAULT_VDW = 1.60;

/** Covalent radii in Angstroms (used to size ball-and-stick spheres). */
const ELEMENT_COVALENT = {
  H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, P: 1.07, S: 1.05,
  CL: 1.02, BR: 1.20, I: 1.39, FE: 1.32, ZN: 1.22, NA: 1.66, MG: 1.41,
  CA: 1.76, K: 2.03, MN: 1.39, CU: 1.32, NI: 1.24, CO: 1.26,
};
const DEFAULT_COVALENT = 0.77;

/** Standard amino-acid residue names → treated as protein for cartoon mode. */
const AMINO_ACIDS = new Set([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
  'SEC', 'PYL', 'MSE', 'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN',
]);

/** Distinct colours cycled through per protein chain in cartoon mode. */
const CHAIN_PALETTE = [
  0x4c9be8, 0xe8734c, 0x5ec26a, 0xd15fd1, 0xe8c84c,
  0x4ce8d1, 0xe84c88, 0x9b8ce8, 0x8ce89b, 0xe89b4c,
];

/* Reusable scratch objects to avoid per-call allocation in hot paths. */
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _perp = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _mat4 = new THREE.Matrix4();
const _color = new THREE.Color();
const _YAXIS = new THREE.Vector3(0, 1, 0);

/* =========================================================================
 * Small helpers
 * ========================================================================= */

/** Normalise an element symbol to an uppercase table key. */
function normElement(el) {
  if (!el) return '';
  return String(el).trim().toUpperCase();
}
function elementColor(el) {
  const k = normElement(el);
  return ELEMENT_COLORS[k] ?? DEFAULT_COLOR;
}
function elementVdw(el) {
  const k = normElement(el);
  return ELEMENT_VDW[k] ?? DEFAULT_VDW;
}
function elementCovalent(el) {
  const k = normElement(el);
  return ELEMENT_COVALENT[k] ?? DEFAULT_COVALENT;
}

/** Extract an (i, j) atom-index pair from a loosely-typed interaction entry. */
function pairFromEntry(e) {
  if (!e) return null;
  const a =
    e.atom1Index ?? e.atom1 ?? e.a1 ?? e.donorIndex ?? e.donor ?? e.i;
  const b =
    e.atom2Index ?? e.atom2 ?? e.a2 ?? e.acceptorIndex ?? e.acceptor ?? e.j;
  if (a == null || b == null) return null;
  return [a | 0, b | 0];
}

/* =========================================================================
 * VisualizationEngine
 * ========================================================================= */

export class VisualizationEngine {
  /**
   * @param {HTMLElement} container - element the canvas is appended to.
   * @param {object} [options]
   * @param {number} [options.background=0x0d0f14]
   * @param {number} [options.ballStickAtomScale=0.32]
   * @param {number} [options.bondRadius=0.14]
   * @param {number} [options.spaceFillScale=1.0]
   * @param {number} [options.cartoonRadius=0.32]
   * @param {number} [options.trajectoryFps=15]
   * @param {boolean}[options.interpolate=true]  - lerp between MD frames.
   * @param {boolean}[options.multipleBondLines=true]
   */
  constructor(container, options = {}) {
    if (!container) throw new Error('VisualizationEngine: container required');
    this.container = container;

    // Compatibility: the app constructs `new VisualizationEngine(canvas, structure)`.
    // If the second arg looks like a structure (has an atoms array) rather than
    // an options object, treat it as a pending structure and use default options.
    let pendingStructure = null;
    if (options && Array.isArray(options.atoms)) {
      pendingStructure = options;
      options = {};
    }

    this.opts = Object.assign(
      {
        background: 0x0d0f14,
        ballStickAtomScale: 0.32,
        bondRadius: 0.14,
        spaceFillScale: 1.0,
        cartoonRadius: 0.32,
        trajectoryFps: 15,
        interpolate: true,
        multipleBondLines: true,
        atomSegments: 20,
        bondSegments: 12,
      },
      options
    );

    // Global atom registry: flat list across all loaded molecules.
    // Each entry: { element, color:THREE.Color, position:THREE.Vector3,
    //               resName, resSeq, chainId, name, isProtein, source }
    this.atoms = [];
    this.bonds = []; // { a, b, type } with GLOBAL atom indices
    this._receptorAtomCount = 0;

    // Render groups let view modes toggle protein vs. hetero independently.
    this._groups = []; // array of render-group descriptors
    this._cartoonGroup = null; // THREE.Group of tube meshes
    this._interactionGroup = null; // THREE.Group of dashed lines
    this._interactionLines = []; // { line, a, b } for trajectory updates
    // Additive interaction store so drawHydrogenBonds / drawElectrostatic can be
    // called separately without wiping each other.
    this._interactionData = { hydrogenBonds: [], electrostatic: [], hydrophobic: [] };
    this._appendedFrames = 0; // count of streamed trajectory frames

    // Trajectory state.
    this.trajectory = null;
    this._frameIndex = 0;
    this._playing = false;
    this._frameClock = 0; // seconds accumulator for fps pacing
    this._basePositions = null; // Float positions before trajectory applied

    this.viewMode = 'ball-and-stick';
    this._disposed = false;

    this._initScene();
    this._initLights();
    this._initControls();
    this._bindResize();
    this._clock = new THREE.Clock();
    this._loop = this._loop.bind(this);
    this._rafId = requestAnimationFrame(this._loop);

    if (pendingStructure) this.loadStructure(pendingStructure);
  }

  /* ----------------------------------------------------------------------
   * Scene / renderer / camera
   * -------------------------------------------------------------------- */

  _initScene() {
    const w = this.container.clientWidth || 640;
    const h = this.container.clientHeight || 480;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.opts.background);

    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 5000);
    this.camera.position.set(0, 0, 40);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);

    // Root that holds all molecular geometry (so we can recentre easily).
    this.root = new THREE.Group();
    this.scene.add(this.root);
  }

  _initLights() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    this.scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(1, 1, 1);
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0xbfd4ff, 0.35);
    fill.position.set(-1, -0.5, -1);
    this.scene.add(fill);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x202028, 0.4);
    this.scene.add(hemi);
  }

  /* ----------------------------------------------------------------------
   * Trackball-style camera controls (OrbitControls math, inlined).
   * Left drag: rotate. Scroll: zoom. Right drag: pan.
   * -------------------------------------------------------------------- */

  _initControls() {
    this.target = new THREE.Vector3(0, 0, 0);
    this._spherical = new THREE.Spherical();
    this._sphericalDelta = new THREE.Spherical(0, 0, 0);
    this._panOffset = new THREE.Vector3();
    this._zoomScale = 1;
    this._damping = 0.12;
    this._minDistance = 2;
    this._maxDistance = 2000;
    this._rotateSpeed = 1.0;
    this._panSpeed = 1.0;
    this._zoomSpeed = 1.0;

    this._pointer = { x: 0, y: 0, active: null }; // active: 'rotate'|'pan'|null

    const dom = this.renderer.domElement;
    dom.style.touchAction = 'none';

    this._onContextMenu = (e) => e.preventDefault();
    this._onPointerDown = (e) => {
      dom.setPointerCapture?.(e.pointerId);
      this._pointer.x = e.clientX;
      this._pointer.y = e.clientY;
      // button 0 = left → rotate, 2 = right → pan, 1 = middle → pan.
      this._pointer.active = e.button === 0 ? 'rotate' : 'pan';
    };
    this._onPointerMove = (e) => {
      if (!this._pointer.active) return;
      const dx = e.clientX - this._pointer.x;
      const dy = e.clientY - this._pointer.y;
      this._pointer.x = e.clientX;
      this._pointer.y = e.clientY;
      const h = dom.clientHeight || 1;
      if (this._pointer.active === 'rotate') {
        this._sphericalDelta.theta -= (2 * Math.PI * dx / h) * this._rotateSpeed;
        this._sphericalDelta.phi -= (2 * Math.PI * dy / h) * this._rotateSpeed;
      } else {
        this._pan(dx, dy);
      }
    };
    this._onPointerUp = (e) => {
      dom.releasePointerCapture?.(e.pointerId);
      this._pointer.active = null;
    };
    this._onWheel = (e) => {
      e.preventDefault();
      const factor = Math.pow(0.95, this._zoomSpeed);
      if (e.deltaY < 0) this._zoomScale /= factor;
      else this._zoomScale *= factor;
    };

    dom.addEventListener('contextmenu', this._onContextMenu);
    dom.addEventListener('pointerdown', this._onPointerDown);
    dom.addEventListener('pointermove', this._onPointerMove);
    dom.addEventListener('pointerup', this._onPointerUp);
    dom.addEventListener('pointercancel', this._onPointerUp);
    dom.addEventListener('wheel', this._onWheel, { passive: false });
  }

  /** Translate the orbit target in the camera's screen plane. */
  _pan(dx, dy) {
    const dom = this.renderer.domElement;
    _v1.copy(this.camera.position).sub(this.target);
    let distance = _v1.length();
    // account for perspective fov so screen-space pan matches world scale.
    distance *= Math.tan((this.camera.fov / 2) * THREE.MathUtils.DEG2RAD);
    const h = dom.clientHeight || 1;
    const panX = (2 * dx * distance) / h * this._panSpeed;
    const panY = (2 * dy * distance) / h * this._panSpeed;

    _v1.setFromMatrixColumn(this.camera.matrix, 0); // camera right
    this._panOffset.addScaledVector(_v1, -panX);
    _v2.setFromMatrixColumn(this.camera.matrix, 1); // camera up
    this._panOffset.addScaledVector(_v2, panY);
  }

  /** Apply accumulated rotation/zoom/pan with damping (called each frame). */
  _updateControls() {
    _v1.copy(this.camera.position).sub(this.target);
    this._spherical.setFromVector3(_v1);

    this._spherical.theta += this._sphericalDelta.theta;
    this._spherical.phi += this._sphericalDelta.phi;

    // clamp polar angle to avoid gimbal flip.
    const EPS = 1e-4;
    this._spherical.phi = Math.max(
      EPS,
      Math.min(Math.PI - EPS, this._spherical.phi)
    );
    this._spherical.makeSafe();

    this._spherical.radius *= this._zoomScale;
    this._spherical.radius = Math.max(
      this._minDistance,
      Math.min(this._maxDistance, this._spherical.radius)
    );

    this.target.add(this._panOffset);
    _v1.setFromSpherical(this._spherical);
    this.camera.position.copy(this.target).add(_v1);
    this.camera.lookAt(this.target);

    // damping decay
    const d = 1 - this._damping;
    this._sphericalDelta.theta *= d;
    this._sphericalDelta.phi *= d;
    this._panOffset.multiplyScalar(d);
    this._zoomScale = 1;
  }

  /* ----------------------------------------------------------------------
   * Data loading
   * -------------------------------------------------------------------- */

  /**
   * Load the primary (receptor / protein) structure.
   * Replaces any previously loaded receptor + inhibitor.
   */
  loadStructure(structure) {
    this._clearMolecules();
    this.atoms = [];
    this.bonds = [];
    this._ingest(structure, 'receptor');
    this._receptorAtomCount = this.atoms.length;
    this._rebuildAll();
    this.fitCameraToStructure();
    return this;
  }

  /**
   * Load a small-molecule inhibitor / ligand alongside the receptor.
   * Its atoms are appended after the receptor atoms in the global list.
   */
  loadInhibitor(inhibitor, ligandOptions = {}) {
    this._ingest(inhibitor, 'inhibitor', ligandOptions);
    this._rebuildAll();
    return this;
  }

  /**
   * Convert a raw structure object into global atom/bond registry entries.
   * @param {object} mol
   * @param {'receptor'|'inhibitor'} source
   * @param {object} [ligandOptions] - { carbonColor } to highlight ligand carbons
   *   while keeping CPK colours for heteroatoms (standard ligand convention).
   */
  _ingest(mol, source, ligandOptions = {}) {
    if (!mol || !Array.isArray(mol.atoms)) return;
    const offset = this.atoms.length;
    const carbonColor =
      ligandOptions.carbonColor != null ? new THREE.Color(ligandOptions.carbonColor) : null;

    for (const a of mol.atoms) {
      const el = a.element;
      const isProtein =
        source === 'receptor' && AMINO_ACIDS.has(String(a.resName || '').toUpperCase());
      // Ligand highlighting: recolour only carbons, leave O/N/S/etc. CPK.
      const useColor =
        carbonColor && normElement(el) === 'C'
          ? carbonColor.clone()
          : new THREE.Color(elementColor(el));
      this.atoms.push({
        element: el,
        color: useColor,
        position: new THREE.Vector3(
          a.position?.x ?? 0,
          a.position?.y ?? 0,
          a.position?.z ?? 0
        ),
        charge: a.charge ?? 0,
        resName: a.resName ?? '',
        resSeq: a.resSeq ?? 0,
        chainId: a.chainId ?? a.chainID ?? a.chain ?? 'A',
        name: a.name ?? '',
        isProtein,
        source,
      });
    }

    if (Array.isArray(mol.bonds)) {
      for (const b of mol.bonds) {
        const i = (b.atom1Index ?? b.a1 ?? b.i);
        const j = (b.atom2Index ?? b.a2 ?? b.j);
        if (i == null || j == null) continue;
        this.bonds.push({ a: offset + i, b: offset + j, type: b.type ?? 1 });
      }
    }
  }

  /* ----------------------------------------------------------------------
   * Master rebuild — partitions atoms into render groups and builds meshes.
   * -------------------------------------------------------------------- */

  _rebuildAll() {
    this._clearGroups();

    // Partition atoms into protein vs. hetero index sets.
    const proteinAtoms = [];
    const heteroAtoms = [];
    for (let i = 0; i < this.atoms.length; i++) {
      (this.atoms[i].isProtein ? proteinAtoms : heteroAtoms).push(i);
    }

    // Partition bonds: a bond is "hetero" if either endpoint is non-protein.
    const proteinBonds = [];
    const heteroBonds = [];
    for (const bond of this.bonds) {
      const both =
        this.atoms[bond.a]?.isProtein && this.atoms[bond.b]?.isProtein;
      (both ? proteinBonds : heteroBonds).push(bond);
    }

    if (proteinAtoms.length) {
      this._groups.push(this._createRenderGroup('protein', proteinAtoms, proteinBonds));
    }
    if (heteroAtoms.length) {
      this._groups.push(this._createRenderGroup('hetero', heteroAtoms, heteroBonds));
    }

    this._buildCartoon();
    this.setViewMode(this.viewMode);
  }

  /**
   * Build an instanced atom mesh + instanced bond mesh for a subset of atoms.
   * Returns a descriptor with update methods used by trajectory playback.
   */
  _createRenderGroup(name, atomIndices, bonds) {
    const g = {
      name,
      atomIndices, // global indices, in local order
      bonds,
      atomMesh: null,
      bondMesh: null,
      segmentsPerBond: [], // parallel to bonds: number of split-half segments
      totalSegments: 0,
    };

    // --- Atom instanced mesh ---
    const sphereGeo = new THREE.SphereGeometry(
      1,
      this.opts.atomSegments,
      Math.max(8, this.opts.atomSegments >> 1)
    );
    const sphereMat = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      shininess: 40,
      specular: 0x222222,
    });
    const atomMesh = new THREE.InstancedMesh(sphereGeo, sphereMat, atomIndices.length);
    atomMesh.frustumCulled = false;
    g.atomMesh = atomMesh;
    this.root.add(atomMesh);

    // --- Bond instanced mesh ---
    // Count split-half segments: each bond becomes (2 * numLines) half-cylinders.
    let totalSegments = 0;
    for (const bond of bonds) {
      const lines = this.opts.multipleBondLines
        ? Math.min(3, Math.max(1, bond.type | 0 || 1))
        : 1;
      const segs = lines * 2;
      g.segmentsPerBond.push(segs);
      totalSegments += segs;
    }
    g.totalSegments = totalSegments;

    if (totalSegments > 0) {
      const cylGeo = new THREE.CylinderGeometry(
        1,
        1,
        1,
        this.opts.bondSegments,
        1,
        true
      );
      const cylMat = new THREE.MeshPhongMaterial({
        color: 0xffffff,
        shininess: 30,
        specular: 0x1a1a1a,
      });
      const bondMesh = new THREE.InstancedMesh(cylGeo, cylMat, totalSegments);
      bondMesh.frustumCulled = false;
      g.bondMesh = bondMesh;
      this.root.add(bondMesh);
    }

    // Populate colours (static) and initial transforms.
    this._writeAtomColors(g);
    this._writeBondColors(g);
    this._updateGroupAtoms(g);
    this._updateGroupBonds(g);

    return g;
  }

  /** Write per-instance atom colours once (colours don't change per frame). */
  _writeAtomColors(g) {
    const mesh = g.atomMesh;
    for (let li = 0; li < g.atomIndices.length; li++) {
      mesh.setColorAt(li, this.atoms[g.atomIndices[li]].color);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  /** Write per-instance bond half colours (split-colour by nearest atom). */
  _writeBondColors(g) {
    if (!g.bondMesh) return;
    let s = 0;
    for (let bi = 0; bi < g.bonds.length; bi++) {
      const bond = g.bonds[bi];
      const cA = this.atoms[bond.a].color;
      const cB = this.atoms[bond.b].color;
      const segs = g.segmentsPerBond[bi]; // even; first half → A, second → B
      const lines = segs / 2;
      for (let l = 0; l < lines; l++) {
        g.bondMesh.setColorAt(s++, cA); // half toward atom A
        g.bondMesh.setColorAt(s++, cB); // half toward atom B
      }
    }
    if (g.bondMesh.instanceColor) g.bondMesh.instanceColor.needsUpdate = true;
  }

  /* ----------------------------------------------------------------------
   * Instance transform updates (also used every trajectory frame)
   * -------------------------------------------------------------------- */

  /** Compute the atom sphere radius for the current view mode. */
  _atomRadius(atom) {
    if (this.viewMode === 'space-fill') {
      return elementVdw(atom.element) * this.opts.spaceFillScale;
    }
    // ball-and-stick: scale covalent radius, keep hydrogens visibly smaller.
    return (0.55 + elementCovalent(atom.element)) * this.opts.ballStickAtomScale;
  }

  /** Rewrite all atom instance matrices for a group from current positions. */
  _updateGroupAtoms(g) {
    const mesh = g.atomMesh;
    for (let li = 0; li < g.atomIndices.length; li++) {
      const atom = this.atoms[g.atomIndices[li]];
      const r = this._atomRadius(atom);
      _mat4.compose(
        atom.position,
        _quat.identity(),
        _scale.set(r, r, r)
      );
      mesh.setMatrixAt(li, _mat4);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
  }

  /** Rewrite all bond half-cylinder instance matrices from current positions. */
  _updateGroupBonds(g) {
    if (!g.bondMesh) return;
    const r = this.opts.bondRadius;
    let s = 0;
    for (let bi = 0; bi < g.bonds.length; bi++) {
      const bond = g.bonds[bi];
      const pA = this.atoms[bond.a].position;
      const pB = this.atoms[bond.b].position;
      const segs = g.segmentsPerBond[bi];
      const lines = segs / 2;

      _dir.subVectors(pB, pA);
      const len = _dir.length() || 1e-6;
      _dir.multiplyScalar(1 / len);

      // Perpendicular for multi-bond offset lines.
      this._perpVector(_dir, _perp);
      const spacing = r * 2.6;

      for (let l = 0; l < lines; l++) {
        const off = (l - (lines - 1) / 2) * spacing;
        // Offset endpoints for this parallel line.
        _v1.copy(pA).addScaledVector(_perp, off);
        _v2.copy(pB).addScaledVector(_perp, off);
        _mid.addVectors(_v1, _v2).multiplyScalar(0.5);
        const subR = lines > 1 ? r * 0.6 : r;

        // Half toward A: v1 -> mid
        this._composeCylinder(_v1, _mid, subR);
        g.bondMesh.setMatrixAt(s++, _mat4);
        // Half toward B: mid -> v2
        this._composeCylinder(_mid, _v2, subR);
        g.bondMesh.setMatrixAt(s++, _mat4);
      }
    }
    g.bondMesh.instanceMatrix.needsUpdate = true;
    g.bondMesh.computeBoundingSphere();
  }

  /** Build a cylinder transform (into _mat4) spanning p→q with radius r. */
  _composeCylinder(p, q, r) {
    _dir.subVectors(q, p);
    const len = _dir.length() || 1e-6;
    _mid.addVectors(p, q).multiplyScalar(0.5);
    _quat.setFromUnitVectors(_YAXIS, _dir.multiplyScalar(1 / len));
    _mat4.compose(_mid, _quat, _scale.set(r, len, r));
  }

  /** Any unit vector perpendicular to `d` (written into `out`). */
  _perpVector(d, out) {
    // Choose the axis least aligned with d to avoid degeneracy.
    if (Math.abs(d.x) < 0.9) out.set(1, 0, 0);
    else out.set(0, 1, 0);
    out.crossVectors(d, out).normalize();
    return out;
  }

  /* ----------------------------------------------------------------------
   * Cartoon ribbon (protein backbone trace)
   * -------------------------------------------------------------------- */

  _buildCartoon() {
    this._cartoonGroup = new THREE.Group();
    this.root.add(this._cartoonGroup);

    // Group protein atoms by chain, then choose one trace point per residue.
    const byChain = new Map(); // chainId -> Map(resSeq -> {caPos, atoms:[]})
    for (let i = 0; i < this.atoms.length; i++) {
      const a = this.atoms[i];
      if (!a.isProtein) continue;
      if (!byChain.has(a.chainId)) byChain.set(a.chainId, new Map());
      const resMap = byChain.get(a.chainId);
      if (!resMap.has(a.resSeq)) resMap.set(a.resSeq, { ca: null, sum: new THREE.Vector3(), n: 0 });
      const res = resMap.get(a.resSeq);
      res.sum.add(a.position);
      res.n++;
      // Prefer an explicit alpha-carbon if atom names are available.
      if (String(a.name).toUpperCase() === 'CA') res.ca = a.position;
    }

    let chainIdx = 0;
    for (const [, resMap] of byChain) {
      // Order residues by sequence number.
      const residues = [...resMap.entries()].sort((x, y) => x[0] - y[0]);
      const points = [];
      for (const [, res] of residues) {
        if (res.ca) points.push(res.ca.clone());
        else if (res.n > 0) points.push(res.sum.clone().multiplyScalar(1 / res.n));
      }
      if (points.length < 2) {
        chainIdx++;
        continue;
      }

      const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5);
      const tubular = Math.max(8, points.length * 6);
      const tubeGeo = new THREE.TubeGeometry(
        curve,
        tubular,
        this.opts.cartoonRadius,
        8,
        false
      );
      const mat = new THREE.MeshPhongMaterial({
        color: CHAIN_PALETTE[chainIdx % CHAIN_PALETTE.length],
        shininess: 25,
        specular: 0x222222,
      });
      const tube = new THREE.Mesh(tubeGeo, mat);
      tube.frustumCulled = false;
      this._cartoonGroup.add(tube);
      chainIdx++;
    }
  }

  /* ----------------------------------------------------------------------
   * View mode switching
   * -------------------------------------------------------------------- */

  /**
   * @param {'ball-and-stick'|'space-fill'|'cartoon'} mode
   */
  setViewMode(mode) {
    if (!['ball-and-stick', 'space-fill', 'cartoon'].includes(mode)) {
      console.warn(`VisualizationEngine: unknown view mode "${mode}"`);
      return this;
    }
    this.viewMode = mode;

    const showCartoon = mode === 'cartoon';
    const showBonds = mode !== 'space-fill';

    for (const g of this._groups) {
      const isProtein = g.name === 'protein';
      // In cartoon mode protein is drawn as a ribbon; hetero shown as sticks.
      const showAtoms = showCartoon ? !isProtein : true;
      const showGroupBonds = showCartoon ? !isProtein : showBonds;

      if (g.atomMesh) g.atomMesh.visible = showAtoms;
      if (g.bondMesh) g.bondMesh.visible = showGroupBonds;

      // Re-scale atoms (space-fill vs. ball-and-stick radii differ).
      if (showAtoms) this._updateGroupAtoms(g);
    }

    if (this._cartoonGroup) this._cartoonGroup.visible = showCartoon;
    return this;
  }

  /* ----------------------------------------------------------------------
   * Non-covalent interaction overlays
   * -------------------------------------------------------------------- */

  /**
   * Draw dashed lines for hydrogen bonds (green), electrostatic (red) and
   * hydrophobic (yellow) contacts. Indices reference the global atom list.
   * @param {object} interactions
   */
  setInteractions(interactions) {
    const data = interactions || {};
    this._interactionData = {
      hydrogenBonds: data.hydrogenBonds || data.hbonds || [],
      electrostatic: data.electrostatic || data.electrostaticInteractions || [],
      hydrophobic: data.hydrophobic || data.hydrophobicContacts || [],
    };
    this._rebuildInteractions();
    return this;
  }

  /** Set/replace only the hydrogen-bond overlay (keeps the others). */
  drawHydrogenBonds(list) {
    this._interactionData.hydrogenBonds = this._normalizeInteractionList(list);
    this._rebuildInteractions();
    return this;
  }

  /** Set/replace only the electrostatic overlay (keeps the others). */
  drawElectrostaticInteractions(list) {
    this._interactionData.electrostatic = this._normalizeInteractionList(list);
    this._rebuildInteractions();
    return this;
  }

  /** Set/replace only the hydrophobic overlay (keeps the others). */
  drawHydrophobicInteractions(list) {
    this._interactionData.hydrophobic = this._normalizeInteractionList(list);
    this._rebuildInteractions();
    return this;
  }

  /** Accept either a bare array or a wrapper like { bonds:[...] } / { interactions:[...] }. */
  _normalizeInteractionList(list) {
    if (Array.isArray(list)) return list;
    if (list && Array.isArray(list.bonds)) return list.bonds;
    if (list && Array.isArray(list.interactions)) return list.interactions;
    if (list && Array.isArray(list.contacts)) return list.contacts;
    return [];
  }

  /** Rebuild all dashed interaction lines from the current interaction store. */
  _rebuildInteractions() {
    this._clearInteractions();
    this._interactionGroup = new THREE.Group();
    this.root.add(this._interactionGroup);
    this._interactionLines = [];

    const specs = [
      { key: 'hydrogenBonds', color: 0x2ecc40, dash: 0.4, gap: 0.25 },
      { key: 'electrostatic', color: 0xff4136, dash: 0.5, gap: 0.3 },
      { key: 'hydrophobic', color: 0xffdc00, dash: 0.3, gap: 0.3 },
    ];

    for (const spec of specs) {
      const list = this._interactionData[spec.key];
      if (!Array.isArray(list)) continue;
      for (const entry of list) {
        const ends = this._resolveEndpoints(entry);
        if (!ends) continue;

        const geo = new THREE.BufferGeometry().setFromPoints([ends.p1, ends.p2]);
        const mat = new THREE.LineDashedMaterial({
          color: spec.color,
          dashSize: spec.dash,
          gapSize: spec.gap,
          transparent: true,
          opacity: 0.95,
        });
        const line = new THREE.Line(geo, mat);
        line.computeLineDistances();
        line.frustumCulled = false;
        this._interactionGroup.add(line);
        // a / b are global atom indices when known (so the line follows the
        // trajectory); null when the entry only gave raw coordinates (static).
        this._interactionLines.push({ line, a: ends.a, b: ends.b });
      }
    }
  }

  /**
   * Resolve an interaction entry to two endpoints. Very permissive so it works
   * regardless of how the binding analyzer shapes its output:
   *  - atom indices (into the global list): atom1Index/atom2Index, donor/acceptor, i/j...
   *  - atom objects with a .position
   *  - raw coordinate pairs: {start,end}, {points:[p1,p2]}, {position1,position2}
   * @returns {{p1:THREE.Vector3, p2:THREE.Vector3, a:?number, b:?number}|null}
   */
  _resolveEndpoints(entry) {
    if (!entry) return null;

    // Explicit coordinate pairs first.
    if (Array.isArray(entry.points) && entry.points.length >= 2) {
      const p1 = this._toPoint(entry.points[0]);
      const p2 = this._toPoint(entry.points[1]);
      if (p1 && p2) return { p1, p2, a: null, b: null };
    }
    if (entry.start && entry.end) {
      const p1 = this._toPoint(entry.start);
      const p2 = this._toPoint(entry.end);
      if (p1 && p2) return { p1, p2, a: null, b: null };
    }
    if (entry.position1 && entry.position2) {
      const p1 = this._toPoint(entry.position1);
      const p2 = this._toPoint(entry.position2);
      if (p1 && p2) return { p1, p2, a: null, b: null };
    }

    // Otherwise pull two atom references (index or object).
    const refA =
      entry.atom1 ?? entry.a1 ?? entry.donor ?? entry.from ?? entry.i ??
      entry.atom1Index ?? entry.donorIndex ?? entry.donorAtom ?? entry.proteinAtom;
    const refB =
      entry.atom2 ?? entry.a2 ?? entry.acceptor ?? entry.to ?? entry.j ??
      entry.atom2Index ?? entry.acceptorIndex ?? entry.acceptorAtom ?? entry.ligandAtom;

    const rA = this._resolveAtomRef(refA);
    const rB = this._resolveAtomRef(refB);
    if (rA && rB) return { p1: rA.point, p2: rB.point, a: rA.index, b: rB.index };
    return null;
  }

  /** Resolve an atom reference (index or object) to a point + optional index. */
  _resolveAtomRef(ref) {
    if (ref == null) return null;
    if (typeof ref === 'number') {
      const atom = this.atoms[ref];
      if (!atom) return null;
      return { point: atom.position.clone(), index: ref };
    }
    if (typeof ref === 'object') {
      const p = this._toPoint(ref.position ?? ref);
      if (p) return { point: p, index: null };
    }
    return null;
  }

  /** Coerce {x,y,z} / [x,y,z] / {position:{...}} into a THREE.Vector3. */
  _toPoint(v) {
    if (!v) return null;
    if (Array.isArray(v) && v.length >= 3) return new THREE.Vector3(v[0], v[1], v[2]);
    if (v.position) return this._toPoint(v.position);
    if (typeof v.x === 'number' && typeof v.y === 'number' && typeof v.z === 'number') {
      return new THREE.Vector3(v.x, v.y, v.z);
    }
    return null;
  }

  /** Refresh interaction line endpoints (called on trajectory updates). */
  _updateInteractions() {
    if (!this._interactionLines.length) return;
    for (const rec of this._interactionLines) {
      if (rec.a == null || rec.b == null) continue; // static (coordinate) line
      const pos = rec.line.geometry.attributes.position;
      const pa = this.atoms[rec.a].position;
      const pb = this.atoms[rec.b].position;
      pos.setXYZ(0, pa.x, pa.y, pa.z);
      pos.setXYZ(1, pb.x, pb.y, pb.z);
      pos.needsUpdate = true;
      rec.line.geometry.computeBoundingSphere();
      rec.line.computeLineDistances();
    }
  }

  /* ----------------------------------------------------------------------
   * Trajectory playback
   * -------------------------------------------------------------------- */

  /**
   * Load an MD trajectory. Frame positions map to the global atom list; if a
   * frame's length equals the receptor atom count, only the receptor moves.
   * @param {Array<{step:number, positions:Array<{x,y,z}>}>} trajectory
   */
  loadTrajectory(trajectory) {
    const frames = this._normalizeTrajectory(trajectory);
    this.trajectory = frames.length ? frames : null;
    this._frameIndex = 0;
    this._frameClock = 0;
    this._appendedFrames = this.trajectory ? this.trajectory.length : 0;
    // Snapshot the loaded (frame-0 / static) positions as a baseline.
    this._basePositions = this.atoms.map((a) => a.position.clone());
    if (this.trajectory && this.trajectory.length) this.setFrame(0);
    return this;
  }

  /**
   * Append newly-recorded frames to the playback buffer (used to stream a
   * running simulation into the viewer as "recorded playback"). Accepts the
   * same loose shapes as loadTrajectory.
   */
  appendTrajectory(frames) {
    const norm = this._normalizeTrajectory(frames);
    if (!norm.length) return this;
    if (!this.trajectory) {
      this.trajectory = [];
      this._basePositions = this.atoms.map((a) => a.position.clone());
    }
    for (const f of norm) this.trajectory.push(f);
    this._appendedFrames = this.trajectory.length;
    return this;
  }

  /** Stop playback and restore the pre-trajectory (loaded) positions. */
  resetTrajectory() {
    this._playing = false;
    this._frameIndex = 0;
    this._frameClock = 0;
    this._appendedFrames = 0;
    if (this._basePositions) {
      for (let i = 0; i < this.atoms.length; i++) {
        if (this._basePositions[i]) this.atoms[i].position.copy(this._basePositions[i]);
      }
      for (const g of this._groups) {
        this._updateGroupAtoms(g);
        this._updateGroupBonds(g);
      }
      this._updateInteractions();
    }
    this.trajectory = null;
    return this;
  }

  /**
   * Coerce assorted trajectory shapes into [{ step, positions:[{x,y,z}] }].
   * Handles: an array of frames, or a wrapper { frames|trajectory|steps:[...] };
   * each frame may be { positions:[...] } or a bare array of positions; each
   * position may be {x,y,z} or [x,y,z].
   */
  _normalizeTrajectory(input) {
    if (!input) return [];
    let arr = input;
    if (!Array.isArray(arr)) {
      arr = input.frames || input.trajectory || input.steps || input.data || [];
    }
    if (!Array.isArray(arr)) return [];

    const out = [];
    for (let f = 0; f < arr.length; f++) {
      const frame = arr[f];
      const rawPos = Array.isArray(frame) ? frame : frame && frame.positions;
      if (!Array.isArray(rawPos)) continue;
      const positions = [];
      for (const p of rawPos) {
        if (Array.isArray(p)) positions.push({ x: p[0], y: p[1], z: p[2] });
        else if (p && p.position) positions.push({ x: p.position.x, y: p.position.y, z: p.position.z });
        else if (p) positions.push({ x: p.x, y: p.y, z: p.z });
      }
      out.push({ step: (frame && frame.step) ?? f, positions });
    }
    return out;
  }

  /** Jump to a specific frame index (clamped). */
  setFrame(index) {
    if (!this.trajectory || !this.trajectory.length) return this;
    const n = this.trajectory.length;
    this._frameIndex = ((index % n) + n) % n;
    this._applyFrame(this._frameIndex, 0);
    return this;
  }

  play() {
    // Allow play even with an empty/1-frame buffer: streamed frames may arrive
    // after playback starts (_stepTrajectory no-ops until >= 2 frames exist).
    this._playing = true;
    return this;
  }
  pause() {
    this._playing = false;
    return this;
  }
  get isPlaying() {
    return this._playing;
  }
  get frameIndex() {
    return this._frameIndex;
  }
  get frameCount() {
    return this.trajectory ? this.trajectory.length : 0;
  }

  /**
   * Apply a frame's positions to the global atom list, optionally lerping
   * a fraction `t` toward the next frame for smooth playback.
   */
  _applyFrame(index, t) {
    const frame = this.trajectory[index];
    if (!frame || !Array.isArray(frame.positions)) return;
    const positions = frame.positions;
    const count = Math.min(positions.length, this.atoms.length);

    let next = null;
    if (t > 0 && this.opts.interpolate) {
      const nf = this.trajectory[(index + 1) % this.trajectory.length];
      if (nf && Array.isArray(nf.positions)) next = nf.positions;
    }

    for (let i = 0; i < count; i++) {
      const p = positions[i];
      if (!p) continue;
      if (next && next[i]) {
        const q = next[i];
        this.atoms[i].position.set(
          p.x + (q.x - p.x) * t,
          p.y + (q.y - p.y) * t,
          p.z + (q.z - p.z) * t
        );
      } else {
        this.atoms[i].position.set(p.x, p.y, p.z);
      }
    }

    // Push updated positions into all instanced meshes + overlays.
    for (const g of this._groups) {
      this._updateGroupAtoms(g);
      this._updateGroupBonds(g);
    }
    this._updateInteractions();
    // Note: cartoon ribbon is not re-splined per frame for performance.
  }

  /** Advance playback based on elapsed real time and configured fps. */
  _stepTrajectory(dt) {
    if (!this._playing || !this.trajectory || this.trajectory.length < 2) return;
    this._frameClock += dt;
    const frameDur = 1 / Math.max(1, this.opts.trajectoryFps);
    const t = Math.min(1, this._frameClock / frameDur);
    this._applyFrame(this._frameIndex, t);
    if (this._frameClock >= frameDur) {
      this._frameClock -= frameDur;
      this._frameIndex = (this._frameIndex + 1) % this.trajectory.length;
    }
  }

  /* ----------------------------------------------------------------------
   * Camera framing utilities
   * -------------------------------------------------------------------- */

  /** Compute a bounding sphere over all atoms. */
  _computeBounds() {
    const box = new THREE.Box3();
    if (!this.atoms.length) {
      box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    } else {
      for (const a of this.atoms) box.expandByPoint(a.position);
    }
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    if (!(sphere.radius > 0)) sphere.radius = 10;
    return sphere;
  }

  /** Frame the whole structure in view. */
  fitCameraToStructure() {
    const sphere = this._computeBounds();
    this.target.copy(sphere.center);
    const fov = this.camera.fov * THREE.MathUtils.DEG2RAD;
    const dist = (sphere.radius * 1.4) / Math.sin(fov / 2);
    _v1.set(0, 0, 1).multiplyScalar(dist);
    this.camera.position.copy(this.target).add(_v1);
    this.camera.near = Math.max(0.1, dist - sphere.radius * 3);
    this.camera.far = dist + sphere.radius * 6;
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(this.target);
    this._maxDistance = dist * 6;
    return this;
  }

  /** Reset camera to the default framed view. */
  resetView() {
    this._sphericalDelta.set(0, 0, 0);
    this._panOffset.set(0, 0, 0);
    this._zoomScale = 1;
    this.fitCameraToStructure();
    return this;
  }

  /* ----------------------------------------------------------------------
   * Compatibility aliases — method names the application controller calls.
   * These map the app's expected surface onto the engine's native API so the
   * module drops into the existing app without editing every call site.
   * -------------------------------------------------------------------- */

  /** Alias for loadStructure(structure). */
  renderStructure(structure) {
    return this.loadStructure(structure);
  }

  /**
   * Alias for loadInhibitor(). The app passes an orange highlight colour; we
   * apply it to the ligand's carbons only and keep CPK colours for heteroatoms
   * (the standard way to make a bound ligand read as a distinct molecule).
   * @param {object} inhibitor
   * @param {number} [color=0xffaa00]
   */
  renderLigand(inhibitor, color = 0xffaa00) {
    return this.loadInhibitor(inhibitor, { carbonColor: color });
  }

  /** Alias for fitCameraToStructure(). */
  zoomToFit() {
    return this.fitCameraToStructure();
  }

  /** Alias for resetView(). */
  resetCamera() {
    return this.resetView();
  }

  /* ----------------------------------------------------------------------
   * Picking (optional convenience): return atom index under the pointer.
   * -------------------------------------------------------------------- */

  /**
   * Raycast from a client-space (x, y) point against atom spheres.
   * @returns {number|null} global atom index or null.
   */
  pickAtom(clientX, clientY) {
    const dom = this.renderer.domElement;
    const rect = dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    if (!this._raycaster) this._raycaster = new THREE.Raycaster();
    this._raycaster.setFromCamera(ndc, this.camera);

    let best = null;
    let bestDist = Infinity;
    for (const g of this._groups) {
      if (!g.atomMesh || !g.atomMesh.visible) continue;
      const hits = this._raycaster.intersectObject(g.atomMesh, false);
      for (const h of hits) {
        if (h.distance < bestDist && h.instanceId != null) {
          bestDist = h.distance;
          best = g.atomIndices[h.instanceId];
        }
      }
    }
    return best;
  }

  /* ----------------------------------------------------------------------
   * Render loop
   * -------------------------------------------------------------------- */

  _loop() {
    if (this._disposed) return;
    const dt = Math.min(0.1, this._clock.getDelta());
    this._stepTrajectory(dt);
    this._updateControls();
    this.renderer.render(this.scene, this.camera);
    this._rafId = requestAnimationFrame(this._loop);
  }

  /** Force a single render (useful if the RAF loop is externally paused). */
  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /* ----------------------------------------------------------------------
   * Resize handling
   * -------------------------------------------------------------------- */

  _bindResize() {
    this._onResize = () => {
      const w = this.container.clientWidth || 640;
      const h = this.container.clientHeight || 480;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    };
    if (typeof ResizeObserver !== 'undefined') {
      this._resizeObserver = new ResizeObserver(this._onResize);
      this._resizeObserver.observe(this.container);
    } else {
      window.addEventListener('resize', this._onResize);
    }
  }

  /* ----------------------------------------------------------------------
   * Teardown helpers
   * -------------------------------------------------------------------- */

  _disposeObject3D(obj) {
    obj.traverse?.((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
        else child.material.dispose();
      }
    });
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
      else obj.material.dispose();
    }
  }

  _clearGroups() {
    for (const g of this._groups) {
      if (g.atomMesh) {
        this.root.remove(g.atomMesh);
        this._disposeObject3D(g.atomMesh);
      }
      if (g.bondMesh) {
        this.root.remove(g.bondMesh);
        this._disposeObject3D(g.bondMesh);
      }
    }
    this._groups = [];
    if (this._cartoonGroup) {
      this.root.remove(this._cartoonGroup);
      this._disposeObject3D(this._cartoonGroup);
      this._cartoonGroup = null;
    }
  }

  _clearInteractions() {
    if (this._interactionGroup) {
      this.root.remove(this._interactionGroup);
      this._disposeObject3D(this._interactionGroup);
      this._interactionGroup = null;
    }
    this._interactionLines = [];
  }

  _clearMolecules() {
    this._clearGroups();
    this._clearInteractions();
    this.trajectory = null;
    this._playing = false;
    this._frameIndex = 0;
  }

  /** Fully tear down the engine: stop RAF, remove listeners, free GPU memory. */
  dispose() {
    this._disposed = true;
    if (this._rafId) cancelAnimationFrame(this._rafId);

    const dom = this.renderer.domElement;
    dom.removeEventListener('contextmenu', this._onContextMenu);
    dom.removeEventListener('pointerdown', this._onPointerDown);
    dom.removeEventListener('pointermove', this._onPointerMove);
    dom.removeEventListener('pointerup', this._onPointerUp);
    dom.removeEventListener('pointercancel', this._onPointerUp);
    dom.removeEventListener('wheel', this._onWheel);

    if (this._resizeObserver) this._resizeObserver.disconnect();
    else window.removeEventListener('resize', this._onResize);

    this._clearMolecules();
    this.renderer.dispose();
    if (dom.parentNode) dom.parentNode.removeChild(dom);
  }
}

export default VisualizationEngine;
