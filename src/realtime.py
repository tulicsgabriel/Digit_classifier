#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 15:52:13 2025

@author: gabriel
"""

#!/usr/bin/env python3
import argparse, time, sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import joblib
import librosa


# -------- util --------
def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def record_once(seconds: float, sr: int) -> np.ndarray:
    """Record a single-channel float32 clip for `seconds` at samplerate `sr`."""
    print(f"🎙️  Recording {seconds:.2f}s... (sr={sr})")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", blocking=False)
    sd.wait()
    return audio[:, 0].copy()


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence; fall back to original if empty."""
    if y.size == 0:
        return y
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trim if y_trim.size > 0 else y


def features_from_signal(fe, y: np.ndarray, sr_in: int) -> np.ndarray:
    """Apply saved MFCCStatsExtractor to a raw signal (resample->frames->CMVN->pool)."""
    if sr_in != fe.sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=fe.sr)
    # Light normalization to avoid ultra-quiet captures
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-8)
    y = trim_silence(y, top_db=30.0)

    # Reuse the extractor's internals
    F = fe._frame_mfcc_stack(y)  # [D x T]
    F = (F - fe.cmvn_mean[:, None]) / (fe.cmvn_std[:, None] + fe.eps)
    stats = [F.mean(axis=1), F.std(axis=1), F.min(axis=1), F.max(axis=1)]
    x = np.concatenate(stats, axis=0).astype(np.float32)[None, :]  # [1 x 4D]
    return x


def main():
    ap = argparse.ArgumentParser(description="Live microphone digit classifier (0–9).")
    ap.add_argument(
        "--model", type=str, default="model.joblib", help="Path to joblib with {'fe','clf'}."
    )
    ap.add_argument("--mic-sr", type=int, default=16000, help="Microphone capture rate (Hz).")
    ap.add_argument(
        "--seconds", type=float, default=1.0, help="Duration to record per attempt (s)."
    )
    ap.add_argument("--topk", type=int, default=3, help="Show top-k results.")
    args = ap.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(f"Model file not found: {path.resolve()}")
        sys.exit(1)

    bundle = joblib.load(path)
    fe = bundle["fe"]
    clf = bundle["clf"]
    labels = [str(i) for i in range(10)]  # consistent with FSDD label order

    print("Loaded model ✅")
    print(f"- Feature SR: {fe.sr} Hz")
    print(f"- Mic capture SR: {args.mic_sr} Hz")
    print("Press ENTER to record; type 'q' + ENTER to quit.")

    # warmup (allocates buffers, just to be neat)
    _ = np.zeros(int(0.05 * args.mic_sr), dtype=np.float32)

    while True:
        try:
            user = input("\n↩︎ ENTER to record (or q to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            user = "q"
        if user == "q":
            print("Bye!")
            break

        y = record_once(args.seconds, args.mic_sr)

        t0 = time.perf_counter()
        X = features_from_signal(fe, y, sr_in=args.mic_sr)
        t1 = time.perf_counter()
        pred = clf.predict(X)[0]
        # scores: prefer probabilities if available, else softmax(decision_function)
        if hasattr(clf[-1], "predict_proba"):
            probs = clf.predict_proba(X)[0]
        elif hasattr(clf[-1], "decision_function"):
            scores = clf.decision_function(X)[0]
            probs = softmax_np(scores)
        else:
            probs = None
        t2 = time.perf_counter()

        # Report
        print(f"→ Predicted digit: {pred}")
        if probs is not None:
            topk = min(args.topk, len(labels))
            order = np.argsort(probs)[::-1][:topk]
            print("   Top-k:")
            for i in order:
                bar = "▮" * int(round(probs[i] * 20))
                print(f"   {labels[i]}  {probs[i]:.2f} {bar}")
        print(f"Latency: feature { (t1-t0)*1000:.1f} ms | inference { (t2-t1)*1000:.1f} ms")


if __name__ == "__main__":
    main()
