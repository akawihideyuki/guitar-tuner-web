/* 
  Guitar Tuner (Web) — Chrome
  - マイク入力を取得してAudioWorkletでリングバッファに蓄積
  - 自己相関(Auto-correlation)で基本周波数を推定
  - 任意で簡易FFT（AnalyserNode）によるピーク補正（ハイブリッド）
  - A4基準切替、バッファ/ゲート/スムージング調整、テストトーンあり

  ▼自己相関（正規化）の概念（擬似数式）
    r[k] = sum_{n=0}^{N-k-1} x[n] * x[n+k] / sqrt( sum x[n]^2 * sum x[n+k]^2 )
    ここで最大のr[k]となる遅延k*に対し、f0 ≈ Fs / k*
    さらにパラボラ補間：
      k̂ = k* + (r[k*+1] - r[k*-1]) / (2*(2*r[k*] - r[k*-1] - r[k*+1]))

  ・低域（E2=82Hz付近）まで安定しやすい利点。
  ・バッファ長は周波数分解能とレイテンシのトレードオフ（1024〜4096 推奨）
    例: 1024@44.1kHz ≈ 23ms, 2048 ≈ 46ms（より安定）、4096 ≈ 93ms（さらに安定）。
*/

// ===== UI Elements =====
const els = {
  start: document.getElementById('btn-start'),
  stop: document.getElementById('btn-stop'),
  status: document.getElementById('status-text'),
  err: document.getElementById('error-text'),
  note: document.getElementById('note'),
  freq: document.getElementById('freq'),
  cents: document.getElementById('cents'),
  needle: document.getElementById('needle'),
  inrange: document.getElementById('inrange'),
  confBar: document.getElementById('conf-bar'),
  confVal: document.getElementById('conf-val'),
  a4Radios: document.querySelectorAll('input[name="a4"]'),
  bufferSel: document.getElementById('buffer-size'),
  noiseDb: document.getElementById('noise-db'),
  noiseDbVal: document.getElementById('noise-db-val'),
  smooth: document.getElementById('smooth'),
  smoothVal: document.getElementById('smooth-val'),
  hybridFFT: document.getElementById('hybrid-fft'),
  toneOn: document.getElementById('tone-on'),
  toneFreq: document.getElementById('tone-freq'),
  toneFreqVal: document.getElementById('tone-freq-val'),
};

// ===== State =====
let audioCtx = null;
let micStream = null;
let workletNode = null;
let analyser = null; // FFT用（ハイブリッド補正時に利用）
let scriptProcessor = null; // フォールバック
let ringBuffer = null;
let ringWritePos = 0;
let ringSize = 16384;
let frameSize = 2048;
let latestSampleRate = 44100;
let running = false;

let a4 = 440;
let noiseGateDb = -60;
let noiseGateAmp = dbToAmp(noiseGateDb);
let smoothFactor = 0.6; // 0..0.9
let useHybridFFT = false;

let updateTimer = null;
let smoothedFreq = 0;

let osc = null; // test tone

// ===== Helpers =====
function dbToAmp(db){ return Math.pow(10, db / 20); }
function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function log2(x){ return Math.log(x) / Math.log(2); }

// Note mapping
const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function freqToMidi(f, refA=a4){ return 69 + 12 * (Math.log(f/refA)/Math.log(2)); }
function midiToFreq(m, refA=a4){ return refA * Math.pow(2, (m - 69)/12); }
function nearestNoteName(f, refA=a4){
  const midi = Math.round(freqToMidi(f, refA));
  const name = NOTE_NAMES[midi%12];
  const oct = Math.floor(midi/12)-1;
  const ideal = midiToFreq(midi, refA);
  const cents = 1200 * Math.log(f/ideal) / Math.log(2);
  return {name, oct, midi, ideal, cents};
}

// ===== Ring Buffer =====
function initRing(size=16384){
  ringSize = size;
  ringBuffer = new Float32Array(ringSize);
  ringWritePos = 0;
}
function writeToRing(data){
  // data: Float32Array
  const n = data.length;
  if(!ringBuffer) return;
  if(n >= ringSize){
    // keep the last part
    ringBuffer.set(data.subarray(n-ringSize));
    ringWritePos = 0;
    return;
  }
  const endSpace = ringSize - ringWritePos;
  if(n <= endSpace){
    ringBuffer.set(data, ringWritePos);
    ringWritePos += n;
    if(ringWritePos === ringSize) ringWritePos = 0;
  }else{
    ringBuffer.set(data.subarray(0, endSpace), ringWritePos);
    ringBuffer.set(data.subarray(endSpace), 0);
    ringWritePos = n - endSpace;
  }
}
function readLastFrame(N){
  // Return latest N samples in new array (contiguous)
  const out = new Float32Array(N);
  const start = (ringWritePos - N + ringSize) % ringSize;
  if(start + N <= ringSize){
    out.set(ringBuffer.subarray(start, start+N));
  }else{
    const first = ringSize - start;
    out.set(ringBuffer.subarray(start), 0);
    out.set(ringBuffer.subarray(0, N-first), first);
  }
  return out;
}

// ===== Pitch Detection: Auto-correlation (normalized) =====
function autocorrelate(buf, sampleRate){
  const N = buf.length;
  // 1) Pre-check: RMS (ノイズゲート)
  let sumsq = 0;
  for(let i=0;i<N;i++){ const x=buf[i]; sumsq += x*x; }
  const rms = Math.sqrt(sumsq/N);
  if(rms < noiseGateAmp) {
    return { freq: 0, confidence: 0, rms };
  }

  // 2) 正規化自己相関
  //   探索範囲をギター想定：E2(82Hz)〜E4(330Hz) 近辺
  const minLag = Math.floor(sampleRate/1000* (1000/330)); // ≈ sampleRate/330
  const maxLag = Math.min(N-1, Math.floor(sampleRate/70)); // 70Hz程度まで見ておく（余裕）
  let bestLag = -1; 
  let bestR = 0;

  // 事前に総エネルギーを使い回し
  let energy0 = 0;
  for(let i=0;i<N;i++) energy0 += buf[i]*buf[i];

  for(let lag=minLag; lag<=maxLag; lag++){
    let num=0, den0=0, den1=0;
    // 窓は単純（改良余地あり：Hann等）
    for(let i=0;i<N-lag;i++){
      const x0 = buf[i];
      const x1 = buf[i+lag];
      num += x0 * x1;
      den0 += x0 * x0;
      den1 += x1 * x1;
    }
    const den = Math.sqrt(den0 * den1) + 1e-12;
    const r = num / den;
    if(r > bestR){
      bestR = r;
      bestLag = lag;
    }
  }

  if(bestLag <= 0 || !isFinite(bestR)) return { freq:0, confidence:0, rms };

  // 3) パラボラ補間（近傍3点）
  const rAt = (lag)=>{
    let num=0, den0=0, den1=0;
    for(let i=0;i<N-lag;i++){
      const x0 = buf[i];
      const x1 = buf[i+lag];
      num += x0 * x1;
      den0 += x0 * x0;
      den1 += x1 * x1;
    }
    return num / (Math.sqrt(den0*den1)+1e-12);
  };
  let refinedLag = bestLag;
  if(bestLag>1 && bestLag<maxLag){
    const r1 = rAt(bestLag-1);
    const r0 = bestR;
    const r2 = rAt(bestLag+1);
    const denom = (2*(2*r0 - r1 - r2));
    if(Math.abs(denom) > 1e-12){
      const delta = (r2 - r1)/denom;
      refinedLag = bestLag + delta;
      // bestRを更新（近似）
      bestR = r0 - 0.25*(r1 - r2)*delta;
    }
  }

  const freq = sampleRate / refinedLag;
  const confidence = clamp((bestR + rms) / 2, 0, 1); // rとrmsを簡易合成
  return { freq, confidence, rms };
}

// ===== Hybrid FFT refinement (optional) =====
function refineWithFFT(baseFreq){
  // AnalyserNodeからパワースペクトルを取得して近傍を二次補間
  if(!analyser || !baseFreq || baseFreq<=0) return baseFreq;
  const fftSize = analyser.fftSize; // 2048 など
  const binCount = analyser.frequencyBinCount;
  const freqs = new Float32Array(binCount);
  analyser.getFloatFrequencyData(freqs); // dBスケール（負の値）

  // 一番高いピークを単純探索（改良余地あり：baseFreq±半音で絞る など）
  // ここでは baseFreq 近傍 ± 20% に制限
  const nyquist = latestSampleRate / 2;
  const toFreq = (bin)=> bin * nyquist / binCount;
  const toBin = (f)=> Math.round(f * binCount / nyquist);

  const low = clamp(toBin(baseFreq*0.8), 1, binCount-2);
  const high = clamp(toBin(baseFreq*1.2), 1, binCount-2);

  let maxBin = low;
  let maxVal = -Infinity;
  for(let b=low;b<=high;b++){
    const v = freqs[b];
    if(v > maxVal){
      maxVal = v;
      maxBin = b;
    }
  }
  // 二次補間
  const yl = freqs[maxBin-1], y0 = freqs[maxBin], yr = freqs[maxBin+1];
  const denom = (yl - 2*y0 + yr);
  let binInterp = maxBin;
  if(Math.abs(denom) > 1e-3){
    const delta = 0.5 * (yl - yr) / denom;
    binInterp = maxBin + delta;
  }
  const refined = toFreq(binInterp);
  // 時間領域の結果と差が大きすぎる場合は無視（ロバスト性）
  if(Math.abs(refined - baseFreq) / baseFreq > 0.2) return baseFreq;
  return refined;
}

// ===== UI Update =====
function updateUI(freq, confidence){
  if(!isFinite(freq) || freq<=0){
    els.note.textContent = '--';
    els.freq.textContent = '0.00';
    els.cents.textContent = '0';
    els.needle.style.transform = `translateX(-50%) rotate(0deg)`;
    els.inrange.style.opacity = 0;
  }else{
    const {name, oct, ideal, cents} = nearestNoteName(freq, a4);
    els.note.textContent = `${name}${oct}`;
    els.freq.textContent = freq.toFixed(2);
    const centsClamped = clamp(Math.round(cents), -50, 50);
    els.cents.textContent = (cents>=0?'+':'') + Math.round(cents);
    // 針（±50cents -> ±45deg）
    const deg = (centsClamped/50) * 45;
    els.needle.style.transform = `translateX(-50%) rotate(${deg}deg)`;
    // 合ってればグリーン点灯（±5c）
    els.inrange.style.opacity = Math.abs(cents) <= 5 ? 1 : 0;
    els.needle.style.background = Math.abs(cents) <= 5 ? 'var(--good)' : 'var(--warn)';
  }
  const pc = Math.round(confidence*100);
  els.confBar.style.width = `${pc}%`;
  els.confVal.textContent = `${pc}%`;
}

// ===== Start / Stop =====
async function start(){
  if(running) return;
  try{
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    latestSampleRate = audioCtx.sampleRate;
    initRing(16384);

    // Test tone chain (off initially)
    if(els.toneOn.checked) startTone();
    else stopTone();

    // マイク取得（HTTPS/localhost必須）
    micStream = await navigator.mediaDevices.getUserMedia({audio: {
      echoCancellation: false, noiseSuppression: false, autoGainControl: false
    }});
    els.err.hidden = true;

    const src = audioCtx.createMediaStreamSource(micStream);

    // Analyser for optional FFT refine
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048; // 2^n
    analyser.smoothingTimeConstant = 0.0;
    src.connect(analyser);

    // Try AudioWorklet
    let workletOK = false;
    if (audioCtx.audioWorklet) {
      try{
        const url = await buildAndLoadWorkletModule();
        workletNode = new AudioWorkletNode(audioCtx, 'tuner-processor', { 
          processorOptions: { frameSize },
        });
        workletNode.port.onmessage = (ev)=>{
          if(ev.data?.type === 'chunk'){
            const arr = new Float32Array(ev.data.buffer);
            writeToRing(arr);
          }else if(ev.data?.type === 'sr'){
            latestSampleRate = ev.data.sampleRate || latestSampleRate;
          }
        };
        src.connect(workletNode);
        // Workletは音声出力しないのでdestinationへは接続しない
        workletOK = true;
      }catch(e){
        console.warn('AudioWorklet load failed, fallback to ScriptProcessor', e);
      }
    }

    if(!workletOK){
      // Fallback: ScriptProcessor (deprecatedだが互換性のため)
      const buf = parseInt(els.bufferSel.value,10) || 2048;
      scriptProcessor = audioCtx.createScriptProcessor(buf, 1, 1);
      scriptProcessor.onaudioprocess = (e)=>{
        const ch = e.inputBuffer.getChannelData(0);
        writeToRing(ch);
      };
      src.connect(scriptProcessor);
      scriptProcessor.connect(audioCtx.destination); // 一部ブラウザで動作安定のため
    }

    // periodic update
    running = true;
    els.start.disabled = true;
    els.stop.disabled = false;
    els.status.textContent = `実行中（sampleRate=${latestSampleRate}Hz）`;

    if(updateTimer) clearInterval(updateTimer);
    updateTimer = setInterval(tick, 60); // 60〜80ms間隔
  }catch(err){
    console.error(err);
    els.err.hidden = false;
    els.err.textContent = `マイク取得に失敗しました: ${err.message || err}`;
    els.status.textContent = `HTTPS(またはlocalhost)でアクセスし、ブラウザのマイク許可を確認してください。`;
  }
}

function stop(){
  running = false;
  els.start.disabled = false;
  els.stop.disabled = true;
  if(updateTimer){ clearInterval(updateTimer); updateTimer = null; }
  if(workletNode){ try{ workletNode.port.postMessage({type:'close'}); }catch{}; workletNode.disconnect(); workletNode = null; }
  if(scriptProcessor){ try{ scriptProcessor.disconnect(); }catch{}; scriptProcessor = null; }
  if(analyser){ try{ analyser.disconnect(); }catch{}; analyser = null; }
  if(micStream){
    micStream.getTracks().forEach(t=>t.stop());
    micStream = null;
  }
  if(audioCtx){ try{ audioCtx.close(); }catch{}; audioCtx = null; }
  stopTone();
  els.status.textContent = '停止しました';
}

// ===== Periodic analysis =====
function tick(){
  // 最新フレーム取得
  const N = frameSize;
  const buf = readLastFrame(N);
  const ac = autocorrelate(buf, latestSampleRate);
  let f = ac.freq;
  if(useHybridFFT) f = refineWithFFT(f);

  // smoothing
  if(f>0 && isFinite(f)){
    if(smoothedFreq === 0) smoothedFreq = f;
    smoothedFreq = smoothFactor*smoothedFreq + (1-smoothFactor)*f;
  }else{
    // 0の場合は徐々に減衰
    smoothedFreq *= 0.9;
    if(smoothedFreq < 1) smoothedFreq = 0;
  }
  updateUI(smoothedFreq, ac.confidence || 0);
}

// ===== AudioWorklet (inline via Blob) =====
async function buildAndLoadWorkletModule(){
  const code = `
    class TunerProcessor extends AudioWorkletProcessor {
      constructor (opts){
        super();
        this._frameSize = (opts?.processorOptions?.frameSize) || 2048;
        this._ring = new Float32Array(this._frameSize);
        this._fill = 0;
        this.port.postMessage({type:'sr', sampleRate: sampleRate});
      }
      process(inputs, outputs, params){
        const input = inputs[0];
        if(input && input[0]){
          const ch = input[0];
          // ch.length は128等のブロック、小刻みに到着
          // 直近frameSize分を集めて送る
          const L = ch.length;
          if(L >= this._frameSize){
            // 大きすぎる場合は末尾のみ
            const slice = ch.subarray(L - this._frameSize);
            const copy = new Float32Array(slice); 
            this.port.postMessage({type:'chunk', buffer: copy.buffer}, [copy.buffer]);
          }else{
            // リング蓄積
            if(this._fill + L < this._frameSize){
              this._ring.set(ch, this._fill);
              this._fill += L;
            }else{
              const remain = this._frameSize - this._fill;
              this._ring.set(ch.subarray(0, remain), this._fill);
              // 完了 -> 送信
              const copy = new Float32Array(this._ring);
              this.port.postMessage({type:'chunk', buffer: copy.buffer}, [copy.buffer]);
              // 次のフレームに残りを詰める
              const rest = L - remain;
              this._ring.fill(0);
              this._ring.set(ch.subarray(remain, remain+rest), 0);
              this._fill = rest;
            }
          }
        }
        return true; // keep alive
      }
    }
    registerProcessor('tuner-processor', TunerProcessor);
  `;
  const blob = new Blob([code], {type:'application/javascript'});
  const url = URL.createObjectURL(blob);
  await audioCtx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  return url;
}

// ===== Test tone =====
function startTone(){
  if(!audioCtx) return;
  if(osc) stopTone();
  osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  gain.gain.value = 0.02; // 小さめ
  osc.frequency.value = parseFloat(els.toneFreq.value)||110;
  osc.connect(gain).connect(audioCtx.destination);
  osc.start();
}
function stopTone(){
  if(osc){ try{ osc.stop(); }catch{}; osc.disconnect(); osc = null; }
}

// ===== Events =====
els.start.addEventListener('click', start);
els.stop.addEventListener('click', stop);

els.a4Radios.forEach(r=>r.addEventListener('change', ()=>{
  a4 = parseInt(document.querySelector('input[name="a4"]:checked').value,10);
}));

els.bufferSel.addEventListener('change', ()=>{
  frameSize = parseInt(els.bufferSel.value,10) || 2048;
  // Workletには次回processから反映される（ring読み出しサイズとして使う）
});

els.noiseDb.addEventListener('input', ()=>{
  noiseGateDb = parseInt(els.noiseDb.value,10);
  noiseGateAmp = dbToAmp(noiseGateDb);
  els.noiseDbVal.textContent = `${noiseGateDb} dB`;
});

els.smooth.addEventListener('input', ()=>{
  smoothFactor = parseFloat(els.smooth.value);
  els.smoothVal.textContent = smoothFactor.toFixed(2);
});

els.hybridFFT.addEventListener('change', ()=>{
  useHybridFFT = els.hybridFFT.checked;
});

els.toneOn.addEventListener('change', ()=>{
  if(!audioCtx){ els.toneOn.checked = false; return; }
  if(els.toneOn.checked) startTone(); else stopTone();
});
els.toneFreq.addEventListener('input', ()=>{
  els.toneFreqVal.textContent = `${els.toneFreq.value} Hz`;
  if(osc) osc.frequency.value = parseFloat(els.toneFreq.value)||110;
});

// ===== Init UI defaults =====
(function initUI(){
  els.noiseDbVal.textContent = `${noiseGateDb} dB`;
  els.smoothVal.textContent = smoothFactor.toFixed(2);
  els.toneFreqVal.textContent = `${els.toneFreq.value} Hz`;
})();

/* 
  === ブラウザ互換性メモ ===
  - 想定は Chrome(最新)。Firefox/Safariでも動作する可能性はあるが、
    AudioWorklet/ScriptProcessorの挙動差やマイク許可UIが異なる。
  - ScriptProcessorNodeはDeprecatedだが、フォールバックとして残置。

  === 改善案 ===
  - 窓関数(Hann)やプリフィルタ(HPF 50Hz)導入でロバスト性UP
  - YIN/MPMアルゴへの置換で誤検出低減
  - ハイブリッド手法の強化：時間領域結果±半音の狭帯域FFTを自前計算
  - マルチピーク追跡＋信頼度の時系列平滑（カルマン等）
  - ネイティブ(iOS/Android)実装：より低レイテンシ＆安定
  - 単体テスト：擬似波形（正弦・鋸歯・ノイズ付与）での自動検証
*/
