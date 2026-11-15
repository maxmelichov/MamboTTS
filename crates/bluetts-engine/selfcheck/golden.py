"""Golden output for the Rust engine's start-up self-check (engine-rs/src/engine.rs,
`selfcheck_*`): ONNX Runtime's CPU EP on the release's vector_estimator.onnx
(notmax123/BlueV3-onnx @d0a0270e, BlueV3 v3.0.1, onnx/; ORT 1.23.2), fed the same integer-hash inputs the
engine builds.

    python golden.py <vector_estimator.onnx> [out=golden.bin]

Needs numpy and onnxruntime. The graph takes one conditional row and runs the
CFG pair itself (cfg_scale input); its output is the next latent. Writes, f32
little-endian: case 0 (12 frames): the whole next latent, [144, 12];
case 1 (88 frames): the next latent at frames 0, 1, 44, 85, [144][4].
Regenerate whenever the inputs or the model change, then rebuild (build.sh):
the file is compiled into the wasm.
"""
import sys

import numpy as np
import onnxruntime as ort

CASES = [(12, 5), (88, 9)]
FRAMES = [0, 1, 44, 85]
STEP, TOTAL = 3, 8
CFG = np.float32(4.0)
LAT, TXT, AS = 144, 256, 256
M32 = 0xFFFFFFFF


def value(seed, i):
    h = ((i * 0x9E3779B1) & M32) ^ ((seed * 0x85EBCA77) & M32)
    h ^= h >> 16
    h = (h * 0x7FEB352D) & M32
    h ^= h >> 15
    h = (h * 0x846CA68B) & M32
    h ^= h >> 16
    return np.float32(h / 4294967296.0 * 2.0 - 1.0)


def values(seed, n):
    return np.array([value(seed, i) for i in range(n)], dtype=np.float32)


def inputs(t, tt):
    lm = np.array([1.0 if i + 2 < t else 0.0 for i in range(t)], np.float32)
    x = (values(1, LAT * t).reshape(LAT, t) * lm).astype(np.float32)
    # scaled near the real inputs' size (powers of two: exact), as the engine does
    emb = (values(2, TXT * tt) * np.float32(0.0625)).reshape(TXT, tt)
    tm = np.ones((tt,), np.float32)
    style = (values(4, 50 * AS) * np.float32(0.125)).reshape(50, AS)
    return x, emb, tm, style, lm


def feeds(x, emb, tm, style, lm, step, total, cfg):
    return {
        'noisy_latent': x[None], 'text_emb': emb[None], 'style_ttl': style[None],
        'latent_mask': lm[None, None], 'text_mask': tm[None, None],
        'current_step': np.array([step], np.float32), 'total_step': np.array([total], np.float32),
        'cfg_scale': np.array([cfg], np.float32),
    }


def main():
    model = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'golden.bin'
    sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    parts = []
    for i, (t, tt) in enumerate(CASES):
        y = sess.run(None, feeds(*inputs(t, tt), STEP, TOTAL, CFG))[0][0]   # [144, t]
        parts.append((y if i == 0 else y[:, FRAMES]).astype(np.float32).ravel())
        print(f'case {i}: {t} frames, {tt} tokens, |out| max {abs(y).max():.3f}')
    data = np.concatenate(parts).astype('<f4')
    data.tofile(out)
    print(f'{out}: {data.size} values, {data.nbytes} bytes, onnxruntime {ort.__version__}')


if __name__ == '__main__':
    main()
