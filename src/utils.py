#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 13:55:05 2025

@author: gabriel
"""

import os
from collections import Counter
import librosa.display
import numpy as np


from sklearn.metrics import confusion_matrix
import seaborn as sns


import matplotlib.pyplot as plt

# Adjust font family and font size
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 20})
plt.rcParams["figure.figsize"] = (30, 22)

# import get_features

PATH_PLOT = "./plots/"

os.makedirs(PATH_PLOT, exist_ok=True)


def label_counts(ds):
    counts = Counter(ds["label"])  # counts by integer id
    id2name = {i: name for i, name in enumerate(ds.features["label"].names)}
    return {id2name[i]: counts[i] for i in sorted(counts)}


def durations(ds):
    sr = ds.features["audio"].sampling_rate
    return np.array([len(x["audio"]["array"]) / sr for x in ds])


def plot_wave_and_spectogram(ex, label="", sample_rate=8000):
    # Waveform plot
    plt.figure()
    plt.plot(ex)
    plt.title("Waveform (8 kHz)")
    plt.xlabel("samples")
    plt.ylabel("amp")
    plt.tight_layout()
    sns.set_style("whitegrid")  # Optional: for a cleaner look
    plt.savefig(f"{PATH_PLOT}waveform_{label}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # MFCC plot
    # F = get_features.MFCCStatsExtractor(sr=sample_rate)._frame_mfcc_stack(ex)  # [D x T]
    # plt.figure()
    # librosa.display.specshow(F[:13, :], x_axis="time")
    # plt.title("MFCCs")
    # plt.colorbar()
    # plt.tight_layout()
    # sns.set_style("whitegrid")  # Optional: for a cleaner look
    # plt.savefig(f"{PATH_PLOT}mfccs_{label}.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # Spectogram
    # Compute the short-time Fourier transform (STFT)
    D = librosa.stft(ex, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Spectrogram plot
    plt.figure()
    librosa.display.specshow(S_db, sr=sample_rate, x_axis="time", y_axis="log", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (8 kHz)")
    plt.tight_layout()
    sns.set_style("white")
    plt.savefig(f"{PATH_PLOT}spectrogram_{label}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_and_save_confusion_matrix(y_true, y_pred, labels, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 16},
        cbar=False,
    )
    plt.title("Confusion Matrix", fontsize=24)
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("True", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=0)
    plt.tight_layout()
    sns.set_style("white")
    plt.savefig(f"{PATH_PLOT}{filename}", dpi=300, bbox_inches="tight")
    plt.close()
