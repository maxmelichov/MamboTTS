"""Parity cases for the native examples (examples/parity.rs, examples/batch.rs):
real inputs (text_encoder.onnx over an IPA string, a voice's style_ttl, seeded
noise) and ONNX Runtime CPU's output of the release's vector_estimator.onnx.

    python cases.py <onnx dir> <voice.json> <out dir> [frames,...]

Per length, writes <out>/case_<t>.bin (one run, step 3 of 8, CFG 4) and
<out>/flow_<t>.bin (the 8-step CFG-4 flow from the same noise, [144, t]).
Case file: u32 [t, tt, tr, step, total, cfg bits], then f32 noisy [144, t],
text_emb [256, tt], style [50, 256], latent_mask [t], text_mask [tt], and
ORT's output (the next latent) [144, t].
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

IPA = ("ðə kwˈɪk bɹˈaʊn fˈɑːks ʤˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ, ænd ðɛn ɹˈʌnz hˈoʊm tə ɪts fˈæmɪli. "
       "ʃiː wˈɔːkt əkɹˈɔs ðɪ ˈoʊld wˈʊdən bɹˈɪʤ ænd wˈɑːʧt ðə bˈoʊts dɹˈɪft slˈoʊli bˈaɪ. ") * 4


def main():
    d, voice, out = sys.argv[1], sys.argv[2], sys.argv[3]
    frames = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [23, 40, 60, 73, 100, 137, 159, 222, 380]
    os.makedirs(out, exist_ok=True)
    vocab = json.load(open(os.path.join(d, 'vocab.json')))['char_to_id']
    te = ort.InferenceSession(os.path.join(d, 'text_encoder.onnx'), providers=['CPUExecutionProvider'])
    ve = ort.InferenceSession(os.path.join(d, 'vector_estimator.onnx'), providers=['CPUExecutionProvider'])
    v = json.load(open(voice))
    style = np.array(v['style_ttl']['data'], np.float32).reshape(1, 50, 256)
    for t in frames:
        tt = max(5, min(int(t * 0.55), 290))
        s = '<en>' + IPA[: tt - 9] + '</en>'
        ids = np.array([[vocab.get(c, 0) for c in s]], np.int64)
        tt = ids.shape[1]
        tm = np.ones((1, 1, tt), np.float32)
        emb = te.run(None, {'text_ids': ids, 'style_ttl': style, 'text_mask': tm})[0]
        rng = np.random.default_rng(t)
        lm = np.ones((1, 1, t), np.float32)
        lm[..., t - 1:] = 0 if t % 2 else 1          # odd lengths: one masked frame at the end
        x0 = (rng.standard_normal((1, 144, t)).astype(np.float32) * lm).astype(np.float32)

        def run(x, step):
            return ve.run(None, {
                'noisy_latent': x, 'text_emb': emb, 'style_ttl': style, 'latent_mask': lm, 'text_mask': tm,
                'current_step': np.array([step], np.float32), 'total_step': np.array([8], np.float32),
                'cfg_scale': np.array([4.0], np.float32),
            })[0]

        want = run(x0, 3)
        head = np.array([t, tt, 50, 3, 8, np.float32(4.0).view(np.uint32)], np.uint32)
        with open(os.path.join(out, f'case_{t}.bin'), 'wb') as f:
            f.write(head.astype('<u4').tobytes())
            for a in (x0, emb, style, lm, tm, want):
                f.write(a.astype('<f4').tobytes())
        x = x0
        for step in range(8):
            x = run(x, step)
        x[0].astype('<f4').tofile(os.path.join(out, f'flow_{t}.bin'))
        print(f'{t} frames, {tt} tokens')


if __name__ == '__main__':
    main()
