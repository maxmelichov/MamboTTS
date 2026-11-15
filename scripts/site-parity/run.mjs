#!/usr/bin/env node
/**
 * End-to-end parity: MamboTTS's server synthesis path against bluetts.com's
 * own worker, on the same inputs (cases.json), stage by stage and sample by
 * sample.
 *
 *   MamboTTS  mambotts-server's BlueRuntime::synthesize_streaming, as a
 *             /v1/audio/speech request with no voice and no speed reaches it
 *             (src/runtime/blue/site_parity.rs, run here with cargo).
 *   site      js/onnx-web-worker.js itself, loaded in Node and driven through
 *             its message protocol as js/browser-tts.js drives it: init with
 *             the WASM backend, then synthesize { text, lang, voice: 'shira',
 *             steps: 8, speed: the page's default (1.0 in every language for
 *             BlueV3 v3.0.1; SITE_SPEED overrides), cfgScale: 4, stream }.
 *             Both sides multiply the speed by the voice's speed_scale[lang].
 *             onnxruntime-web is replaced by onnxruntime-node's CPU provider
 *             (the same kernels as ORT's WASM build, natively), the Hub
 *             downloads by the bundle's local files (same sha256), and the
 *             G2P thread by the worker's own in-thread path (simulate.g2pHere).
 *
 * Both sides run ONNX Runtime 1.23.2 on the CPU with the same session
 * options, so the audio is compared exactly; MAX_ABS (default 0) loosens it.
 *
 * Compared per case: the prepared text, the chunk plan (each chunk's
 * phonemes), its token ids, the duration predictor's seconds and the
 * duration after the speed and the voice's speed_scale (float32, exact),
 * latent frames, the samples of
 * every streamed chunk and the whole clip (float32 before the site's int16
 * encode, and the int16 WAV itself against MamboTTS's float audio encoded the
 * page's way).
 *
 * Russian: MamboTTS reads Russian with polyg2p's Russian (the BlueV3
 * reference's). A site that does too (its worker loads polyg2p's ru_*.lex
 * tiers and js/g2p/russian.js, its older Russian-G2P port, is gone) is
 * compared at every stage like the other languages. Against an older site the
 * phonemes differ by design, so a Russian case compares the prepared text and
 * its sentence units only, and says so. SITE_RU=polyg2p|legacy overrides the
 * detection.
 *
 * Steps: a case may set "steps" (the Quality slider, 5-16); both sides run
 * that many flow steps (default 8).
 *
 * Usage (from the repo root):
 *   BLUETTS_SITE=<LightBlue-site checkout> \
 *   MAMBOTTS_BLUE_MODEL_DIR=<a bluev3-3.0.1 bundle (BlueV3-onnx d0a0270e)> \
 *   BLUEV3_RENIKUD=<RenikudPlus model.onnx @679c56c, fp32> \
 *   ORT_NODE=<node_modules/onnxruntime-node, 1.23.2> \
 *   node scripts/site-parity/run.mjs [filter]
 *
 *   SITE_TTS_JSON the site revision's tts.json (default: the bundle's, the
 *                 same file at d0a0270e); the geometry and operating point
 *                 read from it are checked equal to the bundle's below
 *   SITE_SPEED    the page's default speed (default 1.0, v3.0.1)
 *   PARITY_OUT    working directory (default: a temp dir)
 *   SKIP_CARGO=1  reuse PARITY_OUT/mambo.json from an earlier run
 *   MAX_ABS       allowed per-sample difference (default 0: bit-identical)
 * The cargo step needs ORT_LIB_LOCATION/DYLD_LIBRARY_PATH as for any build.
 * Exit code 1 on any difference.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import module from 'node:module';
import { execFileSync } from 'node:child_process';
import { Readable } from 'node:stream';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const filter = process.argv[2] || '';
const need = (name) => {
    const v = process.env[name];
    if (!v) { console.error(`site-parity: set ${name} (see the header of ${path.relative(REPO, fileURLToPath(import.meta.url))})`); process.exit(2); }
    return path.resolve(v.replace(/^~(?=\/)/, os.homedir()));
};
const SITE = need('BLUETTS_SITE');
const BUNDLE = need('MAMBOTTS_BLUE_MODEL_DIR');
const RENIKUD = need('BLUEV3_RENIKUD');
const ORT_NODE = need('ORT_NODE');
const OUT = path.resolve(process.env.PARITY_OUT || fs.mkdtempSync(path.join(os.tmpdir(), 'mambotts-site-parity-')));
const MAX_ABS = Number(process.env.MAX_ABS || 0);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const CASES = path.join(HERE, 'cases.json');
fs.mkdirSync(OUT, { recursive: true });

// ── MamboTTS: the server's synthesis path ───────────────────────────────────

if (!process.env.SKIP_CARGO) {
    console.log('site-parity: MamboTTS (cargo test -p mambotts-server --release site_parity)');
    execFileSync('cargo', ['test', '-p', 'mambotts-server', '--release', 'site_parity', '--', '--ignored', '--nocapture'], {
        cwd: REPO, stdio: 'inherit',
        env: {
            ...process.env, MAMBOTTS_VE_BACKEND: 'ort', MAMBOTTS_PARITY_OUT: OUT, MAMBOTTS_PARITY_RENIKUD: RENIKUD,
            MAMBOTTS_PARITY_CASES: CASES, MAMBOTTS_PARITY_FILTER: filter,
        },
    });
}
const mambo = JSON.parse(fs.readFileSync(path.join(OUT, 'mambo.json'), 'utf8'));

// ── the site's worker, in Node ──────────────────────────────────────────────

const WORKER = path.join(SITE, 'js', 'onnx-web-worker.js');
const workerSrc = fs.readFileSync(WORKER, 'utf8');
const pick = (re, what) => {
    const m = workerSrc.match(re);
    if (!m) throw new Error(`site-parity: ${what} not found in ${WORKER}`);
    return m[1];
};
const ORT_URL_VERSION = pick(/const ORT_VERSION = '([^']+)'/, 'ORT_VERSION');
const MODEL_REV = pick(/const MODEL_REV = '([0-9a-f]+)'/, 'MODEL_REV');
// Which Russian G2P the site runs (see the header).
const SITE_RU_POLYG2P = process.env.SITE_RU
    ? process.env.SITE_RU === 'polyg2p'
    : !fs.existsSync(path.join(SITE, 'js', 'g2p', 'russian.js')) && workerSrc.includes('ru_accents.lex');
console.log(`site-parity: the site reads Russian with ${SITE_RU_POLYG2P ? "polyg2p's Russian: compared at every stage" : 'its own Russian-G2P port: prepared text and sentence units only'}`);
const { HEBREW_MODELS } = await import(pathToFileURL(path.join(SITE, 'js', 'g2p', 'hebrew.js')).href);
if (fs.statSync(RENIKUD).size !== HEBREW_MODELS.fp32.bytes) {
    throw new Error(`site-parity: ${RENIKUD} is not the site's fp32 RenikudPlus (${HEBREW_MODELS.fp32.bytes} bytes)`);
}

// onnxruntime-web from the CDN -> onnxruntime-node's CPU provider. The site
// asks for 'wasm' (and 'webgpu' only with a GPU, which Node has not).
const shim = path.join(OUT, 'ort-shim.mjs');
fs.writeFileSync(shim, `
import { createRequire } from 'node:module';
const ort = createRequire(import.meta.url)(${JSON.stringify(ORT_NODE)});
export const env = ort.env;
export const Tensor = ort.Tensor;
export const InferenceSession = {
    create: (model, opts = {}) => ort.InferenceSession.create(model instanceof Uint8Array ? model : new Uint8Array(model), {
        ...opts,
        executionProviders: ['cpu'],
        // MamboTTS's sessions (blue-rs load_session): 8 intra-op threads, 1 inter-op.
        intraOpNumThreads: 8, interOpNumThreads: 1,
    }),
};
export default { env, Tensor, InferenceSession };
`);
const ortUrl = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_URL_VERSION}/dist/ort.webgpu.min.mjs`;
module.registerHooks({
    resolve: (spec, ctx, next) => (spec === ortUrl ? { url: pathToFileURL(shim).href, shortCircuit: true } : next(spec, ctx)),
});

// The Hub -> the bundle's files (the same bytes: MANIFEST sha256 = the
// worker's FILES list); the site's own assets are file: URLs here.
const HUB = `https://huggingface.co/notmax123/BlueV3-onnx/resolve/${MODEL_REV}/`;
function localFile(url) {
    if (url.startsWith('file:')) return fileURLToPath(url);
    if (url === HEBREW_MODELS.fp32.url) return RENIKUD;
    if (url.startsWith(HUB)) {
        const rel = url.slice(HUB.length);
        // tts.json: the site's revision's copy (SITE_TTS_JSON) when given;
        // otherwise the bundle's, which is the same file when both pin
        // d0a0270e. The site checks the size, and the geometry and operating
        // point either side reads are checked equal below.
        if (rel === 'onnx/tts.json') return process.env.SITE_TTS_JSON || path.join(BUNDLE, 'tts.json');
        if (rel.startsWith('onnx/')) return path.join(BUNDLE, rel.slice(5));
        if (rel.startsWith('frontend/polyg2p_data/')) return path.join(BUNDLE, 'polyg2p', rel.slice('frontend/polyg2p_data/'.length));
    }
    return null;
}
globalThis.fetch = async (input, opts = {}) => {
    const url = String(input && input.url ? input.url : input);
    const file = localFile(url);
    if (!file || !fs.existsSync(file)) throw new Error(`site-parity: no local file for ${url}`);
    const size = fs.statSync(file).size;
    const range = opts.headers && (opts.headers.Range || opts.headers.range);
    const start = range ? Number(/bytes=(\d+)-/.exec(range)[1]) : 0;
    const body = Readable.toWeb(fs.createReadStream(file, { start }));
    const headers = { 'content-length': String(size - start), 'content-type': url.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream' };
    if (range) headers['content-range'] = `bytes ${start}-${size - 1}/${size}`;
    return new Response(body, { status: range ? 206 : 200, headers });
};

{
    // Everything either side reads from tts.json is the same in both copies.
    const a = JSON.parse(fs.readFileSync(path.join(BUNDLE, 'tts.json'), 'utf8'));
    const b = JSON.parse(fs.readFileSync(localFile(`${HUB}onnx/tts.json`), 'utf8'));
    const used = (c) => [c.ae.sample_rate, c.ae.base_chunk_size, c.ttl.chunk_compress_factor, c.ttl.latent_dim, c.blue_release.inference];
    if (!same(used(a), used(b))) throw new Error('site-parity: the two tts.json files differ in what the engines read');
}

// The worker's globals: `self` with its location and postMessage.
const pending = new Map();
let nextId = 1;
globalThis.self = globalThis;
self.location = { href: pathToFileURL(WORKER).href };
self.postMessage = (msg) => {
    const p = pending.get(msg.id);
    if (!p) return;
    if (msg.type === 'chunk') p.chunks.push(msg);
    else if (msg.type === 'error') { pending.delete(msg.id); p.reject(Object.assign(new Error(msg.message), msg)); }
    else if (p.final.includes(msg.type)) { pending.delete(msg.id); p.resolve({ msg, chunks: p.chunks }); }
};
const call = (type, payload, final) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject, final, chunks: [] });
    self.onmessage({ data: { id, type, payload } });
});

await import(pathToFileURL(WORKER).href);
const Text = await import(pathToFileURL(path.join(SITE, 'js', 'bluetts-text.js')).href);
console.log('site-parity: site worker init (WASM backend = onnxruntime-node CPU)');
const ready = (await call('init', { options: { prefer: 'wasm', simulate: { g2pHere: true } } }, ['ready'])).msg;
console.log(`site-parity: site ready: backend ${ready.backend} (${ready.reason}), Hebrew G2P ${ready.env && ready.env.hebrewG2P && ready.env.hebrewG2P.variant}`);
if (mambo.renikud && ready.env && ready.env.hebrewG2P && ready.env.hebrewG2P.variant !== 'fp32') throw new Error('site-parity: the site did not pick fp32 RenikudPlus');

// The page's request (js/script.js snapshotRequest, js/browser-tts.js synthesize):
// BlueV3 v3.0.1's default speed is 1.0 in every language.
const SITE_SPEED = Number(process.env.SITE_SPEED || 1.0);
if (MODEL_REV !== 'd0a0270e60b96d5066cfe7ebed1f7c2872361c43') {
    console.log(`site-parity: WARNING the site's worker pins BlueV3-onnx ${MODEL_REV}, not v3.0.1 (d0a0270e)`);
}
// The release's fifteen voices, each its own file (no aliases).
const VOICE_MAP = Object.fromEntries(
    ['Maya', 'Noa', 'Lily', 'Tamar', 'Shira', 'Yael', 'Emma', 'Sara', 'Adam', 'Daniel', 'David', 'Jonathan', 'Ethan', 'Michael', 'Ariel']
        .map((name) => [name, name.toLowerCase()]),
);

// ── compare ─────────────────────────────────────────────────────────────────

const f32 = Math.fround;
const toPcm16 = (v) => Math.round(Math.max(-1, Math.min(1, v)) * 32767);
function audioDiff(a, b) {
    if (a.length !== b.length) return { length: [a.length, b.length], maxAbs: Infinity };
    let maxAbs = 0, diff = 0;
    for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] - b[i]);
        if (d > 0) diff++;
        if (d > maxAbs) maxAbs = d;
    }
    return { maxAbs, differing: diff };
}

// heb-tts-normalizer as the worker loads it (js/hebnorm/, the site's wasm build).
let hebnormMod = null;
async function hebnorm() {
    if (!hebnormMod) {
        const mod = await import(pathToFileURL(path.join(SITE, 'js', 'hebnorm', 'hebnorm.js')).href);
        await mod.default({ module_or_path: fs.readFileSync(path.join(SITE, 'js', 'hebnorm', 'hebnorm_bg.wasm')) });
        hebnormMod = mod;
    }
    return hebnormMod;
}

let failures = 0;
const report = [];
for (const m of mambo.cases.filter((c) => c.id.includes(filter))) {
    const lines = [];
    const fail = (msg) => { lines.push(`  FAIL ${msg}`); };
    const lang = m.lang;
    const speed = SITE_SPEED;
    if (Math.abs(m.speed - speed) > 1e-6) fail(`speed: MamboTTS ${m.speed}, site ${speed}`);
    if (VOICE_MAP[m.voice] !== 'shira') fail(`voice: MamboTTS ${m.voice}, site Shira`);
    const heNormalize = lang === 'he' ? async (t) => (await hebnorm()).normalize(t) : null;
    const prepared = await Text.prepareText(m.text, lang, heNormalize);
    if (prepared !== m.prepared) fail(`prepared:\n       mambo ${JSON.stringify(m.prepared)}\n       site  ${JSON.stringify(prepared)}`);
    if (lang === 'ru' && !SITE_RU_POLYG2P) {
        const units = Text.splitSentences(prepared);
        if (units.length !== m.units.length) fail(`Russian sentence units: MamboTTS ${m.units.length}, site ${units.length}`);
        lines.push('  note Russian: the site still has its own Russian-G2P (MamboTTS polyg2p ru); prepared text and sentence units compared only');
        report.push({ id: m.id, ok: !lines.some((l) => l.includes('FAIL')), lines });
        if (lines.some((l) => l.includes('FAIL'))) failures++;
        continue;
    }
    const { msg, chunks: streamed } = await call('synthesize', {
        text: m.text, lang, voice: VOICE_MAP[m.voice], steps: m.steps ?? 8, speed, cfgScale: 4.0, stream: true,
    }, ['audio']);
    const sc = msg.chunks;
    if (sc.length !== m.chunks.length) fail(`chunk count: MamboTTS ${m.chunks.length}, site ${sc.length}`);
    const n = Math.min(sc.length, m.chunks.length);
    for (let k = 0; k < n; k++) {
        const a = m.chunks[k], b = sc[k];
        if (a.phonemes !== b.phonemes) { fail(`chunk ${k} phonemes:\n       mambo ${JSON.stringify(a.phonemes)}\n       site  ${JSON.stringify(b.phonemes)}`); continue; }
        if (!same(a.ids, b.ids)) fail(`chunk ${k} token ids differ`);
        if (f32(a.raw_duration) !== f32(b.rawDur)) fail(`chunk ${k} predicted duration: MamboTTS ${a.raw_duration}, site ${b.rawDur}`);
        if (f32(a.duration) !== f32(b.dur)) fail(`chunk ${k} duration after speed: MamboTTS ${a.duration}, site ${b.dur}`);
        if (a.frames !== b.frames) fail(`chunk ${k} latent frames: MamboTTS ${a.frames}, site ${b.frames}`);
    }
    // Streamed chunks: MamboTTS's are [gap zeros +] chunk, the site's are the
    // chunk with its gap reported separately.
    const raw = fs.readFileSync(path.join(OUT, m.audio));
    const audio = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const site = [];
    for (const c of streamed) {
        for (let i = 0; i < c.gap; i++) site.push(0);
        for (const v of c.pcm) site.push(v);
    }
    const siteAudio = Float32Array.from(site);
    const d = audioDiff(audio, siteAudio);
    if (d.length) fail(`audio length: MamboTTS ${d.length[0]}, site ${d.length[1]} samples`);
    else if (d.maxAbs > MAX_ABS) fail(`audio: ${d.differing} of ${audio.length} samples differ, max |diff| ${d.maxAbs.toExponential(3)}`);
    // The site's saved clip: int16 WAV (44-byte header).
    const wav = new Int16Array(msg.wav.slice(44));
    let pcmDiff = 0;
    if (wav.length === audio.length) for (let i = 0; i < wav.length; i++) pcmDiff = Math.max(pcmDiff, Math.abs(wav[i] - toPcm16(audio[i])));
    else fail(`WAV length: site ${wav.length}, MamboTTS ${audio.length}`);
    if (pcmDiff > Math.ceil(MAX_ABS * 32767)) fail(`int16 WAV: max |diff| ${pcmDiff} LSB`);
    const durs = m.chunks.map((c) => c.duration.toFixed(3)).join(' + ');
    lines.unshift(`  ${m.steps ?? 8} steps, ${n} chunk(s), durations ${durs} s, ${audio.length} samples, audio max |diff| ${Number.isFinite(d.maxAbs) ? d.maxAbs : 'n/a'}`);
    const ok = !lines.some((l) => l.includes('FAIL'));
    if (!ok) failures++;
    report.push({ id: m.id, ok, lines });
}

for (const r of report) {
    console.log(`${r.ok ? 'ok  ' : 'FAIL'} ${r.id}`);
    for (const l of r.lines) console.log(l);
}
console.log(`\nsite-parity: ${report.length - failures} of ${report.length} cases identical (${OUT})`);
// Not process.exit(): onnxruntime-node's thread pools abort the process when
// torn down from under live sessions; the event loop ends on its own.
process.exitCode = failures ? 1 : 0;
