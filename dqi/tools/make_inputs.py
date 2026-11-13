"""Writes DQI test inputs for tools/kmxsim (bit-packed along shots).

usage: python make_inputs.py out.bin --shots 51200 [--seed 1] [--weight W]
Also writes out.bin.weights.npy with the error weight of each shot.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_dqi as V  # noqa: E402


def pack_rows(values, width=V.B):
    """values[shots, regs] -> packed rows [regs*width, shots/8], row = reg*width + bit."""
    shots, regs = values.shape
    out = np.empty((regs * width, shots // 8), dtype=np.uint8)
    v = values.astype(np.uint16)
    for b in range(width):
        out[b::width] = np.packbits(((v >> b) & 1).T.astype(np.uint8), axis=1, bitorder='little')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--shots', type=int, default=51200)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--weight', type=int, default=None)
    ap.add_argument('--chunk', type=int, default=6400)
    a = ap.parse_args()
    assert a.shots % 64 == 0 and a.chunk % 64 == 0
    tab = V.Tables()
    rng = np.random.default_rng(a.seed)
    wp = V.weight_distribution()
    S, E, P, Wt = [], [], [], []
    done = 0
    while done < a.shots:
        n = min(a.chunk, a.shots - done)
        synd, mask, phase, weights = V.draw_batch(tab, rng, n, wp, a.weight)
        S.append(pack_rows(synd))
        E.append(pack_rows(mask))
        P.append(np.packbits(phase, bitorder='little'))
        Wt.append(weights)
        done += n
    with open(a.out, 'wb') as f:
        f.write(np.int64(a.shots).tobytes())
        f.write(np.ascontiguousarray(np.concatenate(S, axis=1)).tobytes())
        f.write(np.ascontiguousarray(np.concatenate(E, axis=1)).tobytes())
        f.write(np.concatenate(P).tobytes())
    w = np.concatenate(Wt)
    np.save(a.out + '.weights.npy', w)
    print('wrote', a.out, 'shots', a.shots, 'mean weight', w.mean())


if __name__ == '__main__':
    main()
