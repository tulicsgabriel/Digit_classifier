#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 13:41:34 2025

@author: gabriel
"""

import numpy as np
import librosa


class MFCCStatsExtractor:
    """
    Waveform -> MFCC(+Δ,+ΔΔ) -> CMVN (fit on train frames) -> [mean,std,min,max] pooling -> fixed-size vector.
    Defaults: 8 kHz, 25 ms window, 10 ms hop, 13 MFCCs (drop c0), +Δ,+ΔΔ -> 36 dims/frame, 144-dim pooled vector.
    """

    def __init__(
        self,
        sr=8000,
        n_mfcc=13,
        drop_c0=True,  # drop MFCC-0 (log-energy) for more loudness invariance
        use_deltas=True,
        use_delta_delta=True,
        win_length_ms=25.0,
        hop_length_ms=10.0,
        n_fft=None,  # if None: next power of 2 >= win_length_samples
        n_mels=40,
        fmin=20.0,
        fmax=None,  # if None: sr/2
        htk=True,  # librosa MFCC defaults to HTK=False; HTK=True is common in ASR
        eps=1e-8,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.drop_c0 = drop_c0
        self.use_deltas = use_deltas
        self.use_delta_delta = use_delta_delta
        self.win_length = int(round(win_length_ms * sr / 1000.0))
        self.hop_length = int(round(hop_length_ms * sr / 1000.0))
        if n_fft is None:
            # next power of 2 >= win_length
            n = 1
            while n < self.win_length:
                n <<= 1
            self.n_fft = n
        else:
            self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2
        self.htk = htk
        self.eps = eps

        # Filled by fit()
        self.cmvn_mean = None  # shape (D,)
        self.cmvn_std = None  # shape (D,)

    # ---------- low-level feature computation ----------
    def _frame_mfcc_stack(self, y):
        """
        Compute MFCC (+Δ, +ΔΔ) frames [D x T].
        """
        # librosa expects float32 mono in [-1,1]; datasets already gives that
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=self.htk,
        )  # [n_mfcc x T]

        if self.drop_c0:
            mfcc = mfcc[1:, :]  # drop coefficient 0 → dims = n_mfcc-1

        feats = [mfcc]
        if self.use_deltas:
            d1 = librosa.feature.delta(mfcc, order=1)  # same shape
            feats.append(d1)
        if self.use_delta_delta:
            d2 = librosa.feature.delta(mfcc, order=2)
            feats.append(d2)

        F = np.vstack(feats)  # [D x T]
        return F

    # ---------- CMVN fit over ALL train frames ----------
    def fit(self, ds_train):
        """
        Accumulate mean/std over all frames from the training split.
        ds_* are HuggingFace datasets with column 'audio' and label 'label'.
        """
        sum_vec, sumsq_vec, n_frames, D = None, None, 0, None

        for ex in ds_train:
            y = ex["audio"]["array"]
            F = self._frame_mfcc_stack(y)  # [D x T]
            if D is None:
                D = F.shape[0]
                sum_vec = np.zeros(D, dtype=np.float64)
                sumsq_vec = np.zeros(D, dtype=np.float64)

            # accumulate across time
            sum_vec += F.sum(axis=1)
            sumsq_vec += (F**2).sum(axis=1)
            n_frames += F.shape[1]

        mean = sum_vec / max(n_frames, 1)
        var = sumsq_vec / max(n_frames, 1) - mean**2
        std = np.sqrt(np.maximum(var, self.eps))

        self.cmvn_mean = mean.astype(np.float32)
        self.cmvn_std = std.astype(np.float32)
        return self

    # ---------- transform split to (X, y) ----------
    def transform(self, ds_split):
        """
        Apply CMVN (using train stats) frame-wise, then pool to fixed vector.
        Returns:
            X: np.ndarray [N x (D * num_stats)]
            y: np.ndarray [N]
        """
        assert self.cmvn_mean is not None, "Call .fit(train_ds) before .transform(...)"
        X, y = [], []
        for ex in ds_split:
            yi = ex["audio"]["array"]
            F = self._frame_mfcc_stack(yi)  # [D x T]
            # CMVN (broadcast across time)
            F = (F - self.cmvn_mean[:, None]) / (self.cmvn_std[:, None] + self.eps)

            # stats pooling over time
            stats = [
                F.mean(axis=1),
                F.std(axis=1),
                F.min(axis=1),
                F.max(axis=1),
            ]  # each [D]
            xi = np.concatenate(stats, axis=0)  # [4D]
            X.append(xi.astype(np.float32))
            y.append(ex["label"])

        X = np.stack(X, axis=0)  # [N x 4D]
        y = np.asarray(y, dtype=np.int64)
        return X, y

    # convenience
    def fit_transform(self, ds_train):
        self.fit(ds_train)
        return self.transform(ds_train)
