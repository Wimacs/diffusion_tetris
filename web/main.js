/* eslint-disable no-undef */
(async function () {
  const cfg = await fetch('./config.json').then(r => r.json());

  // UI elements
  const fileInput = document.getElementById('onnxFile');
  const startImgInput = document.getElementById('startImg');
  const stepsInput = document.getElementById('steps');
  const stepsVal = document.getElementById('stepsVal');
  const scaleInput = document.getElementById('scale');
  const autoplayFpsInput = document.getElementById('autoplayFps');
  const fpsVal = document.getElementById('fpsVal');
  const btnAutoplay = document.getElementById('btnAutoplay');
  const btnStep = document.getElementById('btnStep');
  const btnReset = document.getElementById('btnReset');
  const statusEl = document.getElementById('status');
  const actionEl = document.getElementById('action');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  const W = cfg.generation.image_size;
  const H = cfg.generation.image_size;
  let SCALE = parseInt(scaleInput.value, 10) || 6;

  function setStatus(text) { statusEl.textContent = text; }
  function setAction(text) { if (actionEl) actionEl.textContent = text; }

  // Model/session
  let session = null;
  let isSampling = false; // avoid blocking UI by running multiple samples

  // Keyboard action mapping (same priority as infer_pygame.py)
  let pressed = new Set();
  const ARROW_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);
  window.addEventListener('keydown', (e) => {
    // Prevent inputs/sliders from consuming arrow keys
    if (ARROW_KEYS.has(e.code)) e.preventDefault();
    pressed.add(e.code);
    try {
      const a = getCurrentAction();
      actionStack.push(a);
      setAction(actionName(a));
    } catch {}
    if (e.code === 'KeyA') toggleAutoplay();
    if (e.code === 'KeyS') step();
    if (e.code === 'KeyR') resetSequence();
  }, { capture: true });
  window.addEventListener('keyup', (e) => {
    if (ARROW_KEYS.has(e.code)) e.preventDefault();
    pressed.delete(e.code);
    try {
      const a = getCurrentAction();
      actionStack.push(a);
      setAction(actionName(a));
    } catch {}
  }, { capture: true });

  function getCurrentAction() {
    if (pressed.has('ArrowUp') || pressed.has('Digit1') || pressed.has('Numpad1')) return 1;
    if (pressed.has('ArrowDown') || pressed.has('Digit2') || pressed.has('Numpad2')) return 2;
    if (pressed.has('ArrowLeft') || pressed.has('Digit3') || pressed.has('Numpad3')) return 3;
    if (pressed.has('ArrowRight') || pressed.has('Digit4') || pressed.has('Numpad4')) return 4;
    if (pressed.has('Digit0') || pressed.has('Numpad0')) return 0;
    return 0;
  }

  function actionName(a) {
    if (a === 1) return 'Up';
    if (a === 2) return 'Down';
    if (a === 3) return 'Left';
    if (a === 4) return 'Right';
    return 'No-op';
  }

  // EDM utilities port
  function computeConditioners(sigma, sigmaData, sigmaOffsetNoise) {
    const s2 = sigma * sigma + sigmaOffsetNoise * sigmaOffsetNoise;
    const s = Math.sqrt(s2);
    const c_in = 1 / Math.sqrt(s * s + sigmaData * sigmaData);
    const c_skip = (sigmaData * sigmaData) / (s * s + sigmaData * sigmaData);
    const c_out = s * Math.sqrt(c_skip);
    const c_noise = Math.log(s) / 4;
    return { c_in, c_out, c_skip, c_noise };
  }

  function buildSigmas(numSteps, sigmaMin, sigmaMax, rho) {
    const minInv = Math.pow(sigmaMin, 1 / rho);
    const maxInv = Math.pow(sigmaMax, 1 / rho);
    const sigmas = new Float32Array(numSteps + 1);
    for (let i = 0; i < numSteps; i++) {
      const l = i / (numSteps - 1);
      sigmas[i] = Math.pow(maxInv + l * (minInv - maxInv), rho);
    }
    sigmas[numSteps] = 0;
    return sigmas;
  }

  function tensorToImageData(chwNeg1To1) {
    // chw in [-1,1], shape [3, H, W] -> ImageData RGBA
    const [C, H_, W_] = [3, H, W];
    const out = new ImageData(W_, H_);
    const data = out.data;
    let idx = 0;
    for (let y = 0; y < H_; y++) {
      for (let x = 0; x < W_; x++) {
        const r = chwNeg1To1[0 * H_ * W_ + y * W_ + x];
        const g = chwNeg1To1[1 * H_ * W_ + y * W_ + x];
        const b = chwNeg1To1[2 * H_ * W_ + y * W_ + x];
        data[idx++] = Math.max(0, Math.min(255, Math.round((r + 1) * 127.5)));
        data[idx++] = Math.max(0, Math.min(255, Math.round((g + 1) * 127.5)));
        data[idx++] = Math.max(0, Math.min(255, Math.round((b + 1) * 127.5)));
        data[idx++] = 255;
      }
    }
    return out;
  }

  function drawCHW(chwNeg1To1) {
    const imgData = tensorToImageData(chwNeg1To1);
    const targetW = W * SCALE;
    const targetH = H * SCALE;
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    // nearest-neighbor upscale
    const tmp = document.createElement('canvas');
    tmp.width = W; tmp.height = H;
    tmp.getContext('2d').putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, targetW, targetH);
  }

  // State: prev frames and actions
  const C = cfg.generation.input_channels;
  const CONTEXT = cfg.generation.context_length;
  const ACTIONS = cfg.generation.actions_count;

  let genImgs = null; // list of frames in CHW [-1,1]
  let actionsHist = []; // list of ints
  let startImgCHW = null; // Float32Array length C*H*W in [-1,1]
  let actionStack = []; // pending user actions
  let currentAction = 0; // last action drained from stack

  function randomNormal() {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function resetSequence() {
    // Start from last CONTEXT frames = zeros (or could load one example). We'll use zeros for simplicity.
    genImgs = [];
    for (let i = 0; i < CONTEXT; i++) {
      if (startImgCHW) {
        genImgs.push(startImgCHW.slice());
      } else {
        // Match training normalization (black ~= -1 after (x-0.5)/0.5)
        genImgs.push(new Float32Array(C * H * W).fill(-1));
      }
    }
    actionsHist = new Array(CONTEXT).fill(0);
    actionStack = [];
    currentAction = 0;
    drawCHW(genImgs[genImgs.length - 1]);
    setAction('No-op');
    setStatus('Reset');
  }

  function drainActionStack() {
    if (actionStack.length > 0) {
      currentAction = actionStack[actionStack.length - 1];
      actionStack.length = 0;
      setAction(actionName(currentAction));
    }
  }

  function hwcToChwNorm(hwcUint8) {
    // hwcUint8: Uint8ClampedArray length H*W*4 (RGBA)
    const out = new Float32Array(C * H * W);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const o = (y * W + x) * 4;
        const r = hwcUint8[o];
        const g = hwcUint8[o + 1];
        const b = hwcUint8[o + 2];
        // normalize to [-1,1]
        out[0 * H * W + y * W + x] = (r / 255) * 2 - 1;
        out[1 * H * W + y * W + x] = (g / 255) * 2 - 1;
        out[2 * H * W + y * W + x] = (b / 255) * 2 - 1;
      }
    }
    return out;
  }

  async function loadStartImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        try {
          const tmp = document.createElement('canvas');
          tmp.width = W; tmp.height = H;
          const tctx = tmp.getContext('2d');
          // draw with resize to model size
          tctx.drawImage(img, 0, 0, W, H);
          const imgData = tctx.getImageData(0, 0, W, H);
          startImgCHW = hwcToChwNorm(imgData.data);
          resolve();
        } catch (e) { reject(e); }
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  async function ensureSessionFromFile(file) {
    const buf = await file.arrayBuffer();
    session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    setStatus('ONNX 模型已加载');
  }

  async function ensureSessionFromUrl(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`无法加载模型: ${url}`);
    const buf = await res.arrayBuffer();
    session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    setStatus('ONNX 模型已加载');
  }

  function flattenPrev(noisy, prevFrames, c_in) {
    // noisy: [C,H,W], prevFrames: Array of CONTEXT frames [C,H,W]
    // returns Float32Array of shape [C*(CONTEXT+1), H, W]
    const out = new Float32Array(C * (CONTEXT + 1) * H * W);
    // first is c_in * noisy
    for (let i = 0; i < C * H * W; i++) out[i] = noisy[i] * c_in;
    // then the prev frames
    for (let t = 0; t < CONTEXT; t++) {
      const base = (t + 1) * C * H * W;
      const src = prevFrames[t];
      out.set(src, base);
    }
    return out;
  }

  function wrapModelOutput(noisy, modelOut, cs, quantizeOutput) {
    // per-channel
    const len = noisy.length;
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      let d = cs.c_skip * noisy[i] + cs.c_out * modelOut[i];
      if (quantizeOutput) {
        d = Math.min(1, Math.max(-1, d));
        // Match PyTorch: clamp -> to [0,255] uint8 (floor), then back to [-1,1]
        d = Math.floor(((d + 1) / 2) * 255) / 255 * 2 - 1;
      }
      out[i] = d;
    }
    return out;
  }

  async function denoise(noisy, sigma, prevFrames, prevActions) {
    const cs = computeConditioners(sigma, cfg.edm.sigma_data, cfg.edm.sigma_offset_noise);
    const x = flattenPrev(noisy, prevFrames, cs.c_in);

    // Build ONNX inputs
    // x: [1, C*(CONTEXT+1), H, W]
    // t (c_noise): [1,1]
    // prev_actions: [1, CONTEXT] (int64)
    const xTensor = new ort.Tensor('float32', x, [1, C * (CONTEXT + 1), H, W]);
    const tTensor = new ort.Tensor('float32', new Float32Array([cs.c_noise]), [1, 1]);

    const act64 = new BigInt64Array(CONTEXT);
    for (let i = 0; i < CONTEXT; i++) act64[i] = BigInt(prevActions[i]);
    const actionsTensor = new ort.Tensor('int64', act64, [1, CONTEXT]);

    const feeds = { x: xTensor, t: tTensor, prev_actions: actionsTensor };
    const outputs = await session.run(feeds);
    const y = outputs.y.data; // Float32Array [1,C,H,W]
    const modelOut = y; // flatten [C,H,W]

    return wrapModelOutput(noisy, modelOut, cs, cfg.edm.quantize_output);
  }

  async function sampleOne(steps, prevFrames, prevActions) {
    // x ~ N(0, I)
    let x = new Float32Array(C * H * W);
    for (let i = 0; i < x.length; i++) x[i] = randomNormal();

    const sigmas = buildSigmas(steps, cfg.edm.sigma_min, cfg.edm.sigma_max, cfg.edm.rho);

    for (let i = 0; i < sigmas.length - 1; i++) {
      const sigma = sigmas[i];
      const nextSigma = sigmas[i + 1];

      // karras_step (Euler, order=1)
      // stochasticity (s_churn=0 by default)
      const sigmaHat = sigma * (1 + (cfg.edm.s_churn > 0 && sigma >= cfg.edm.s_tmin && sigma <= (cfg.edm.s_tmax === '.inf' ? Number.POSITIVE_INFINITY : cfg.edm.s_tmax) ? Math.min(cfg.edm.s_churn / Math.max(steps, 1), Math.SQRT2 - 1) : 0));

      const denoised = await denoise(x, sigmaHat, prevFrames, prevActions);
      const inv = 1 / sigmaHat;
      const dt = nextSigma - sigmaHat;
      // single pass update to reduce blocking
      for (let k = 0; k < x.length; k++) x[k] = x[k] + (x[k] - denoised[k]) * inv * dt;

      // yield to the browser so keyboard events remain responsive
      // do it every iteration for smoother UX under low FPS
      // eslint-disable-next-line no-await-in-loop
      await new Promise(requestAnimationFrame);
    }
    return x;
  }

  async function step() {
    if (!session) { setStatus('Please load an ONNX model first'); return; }
    if (isSampling) return;
    isSampling = true;
    try {
      drainActionStack();
      const steps = Math.min(5, Math.max(1, parseInt(stepsInput.value, 10) || 3));

      // actions: append current action
      const act = currentAction;
      setAction(actionName(act));
      actionsHist.push(act);
      if (actionsHist.length > CONTEXT) actionsHist.shift();

      // prev frames: last CONTEXT frames
      const prevFrames = genImgs.slice(-CONTEXT);

      setStatus('Sampling...');
      const nextImg = await sampleOne(steps, prevFrames, actionsHist);
      genImgs.push(nextImg);
      drawCHW(nextImg);
      setStatus('Done');
    } finally {
      isSampling = false;
    }
  }

  let autoplay = true;
  let lastSampleTime = performance.now();
  function toggleAutoplay() {
    autoplay = !autoplay;
    btnAutoplay.textContent = autoplay ? 'Pause Autoplay' : 'Start Autoplay';
  }

  async function loop() {
    // reflect latest user action each frame
    drainActionStack();
    const fps = parseInt(autoplayFpsInput.value, 10) || 5;
    const interval = 1000 / Math.max(1, fps);
    const now = performance.now();
    if (autoplay && !isSampling && now - lastSampleTime >= interval) {
      // launch step without awaiting to keep UI responsive
      // errors will be surfaced to console
      step();
      lastSampleTime = now;
    }
    requestAnimationFrame(loop);
  }

  // Wire UI
  stepsInput.addEventListener('input', () => stepsVal.textContent = stepsInput.value);
  autoplayFpsInput.addEventListener('input', () => fpsVal.textContent = autoplayFpsInput.value);
  scaleInput.addEventListener('change', () => { SCALE = parseInt(scaleInput.value, 10) || 6; drawCHW(genImgs[genImgs.length - 1]); });
  btnAutoplay.addEventListener('click', toggleAutoplay);
  btnStep.addEventListener('click', step);
  btnReset.addEventListener('click', resetSequence);

  fileInput.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    await ensureSessionFromFile(f);
  });

  startImgInput.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setStatus('Loading start image...');
    try {
      await loadStartImage(f);
      resetSequence();
      setStatus('Start image loaded');
    } catch (err) {
      console.error('Failed to load start image', err);
      setStatus('Failed to load start image');
    }
  });

  // Init
  // Ensure page receives keyboard focus so arrow keys are captured
  try { document.body.tabIndex = -1; document.body.focus(); } catch {}

  // Try auto-load start.png if present
  ;(async () => {
    try {
      const startPath = cfg.start_image || './start.png';
      const res = await fetch(startPath);
      if (res.ok) {
        const blob = await res.blob();
        await loadStartImage(new File([blob], 'start.png', { type: blob.type }));
        setStatus('Start image loaded');
      }
    } catch {}
  })();

  // Auto-load ONNX model if configured
  ;(async () => {
    try {
      if (cfg.model_url) {
        await ensureSessionFromUrl(cfg.model_url);
        if (cfg.autoplay === true) autoplay = true;
        setStatus('Model auto-loaded');
      }
    } catch (e) {
      console.error('Failed to auto-load model', e);
    }
  })();
  resetSequence();
  setStatus('Please choose and load an ONNX model');
  loop();
})(); 