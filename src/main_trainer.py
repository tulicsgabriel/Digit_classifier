#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 12:17:29 2025

@author: gabriel
"""

import time
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from datasets import concatenate_datasets

import get_features
import get_audio_data
import utils


if __name__ == "__main__":

    train, val, test = get_audio_data.get_hf_audio_data()

    # %% Peek into the data
    ex = train[0]["audio"]
    print(ex["array"].shape, ex["sampling_rate"])  # (...,), 8000
    print(train.features["label"].names)

    print("sizes:", len(train), len(val), len(test))
    print("train:", utils.label_counts(train))
    print("val:  ", utils.label_counts(val))
    print("test: ", utils.label_counts(test))

    for name, split in [("train", train), ("val", val), ("test", test)]:
        d = utils.durations(split)
        print(
            f"{name}: mean {d.mean():.3f}s, std {d.std():.3f}s, min {d.min():.3f}s, max {d.max():.3f}s"
        )

    # %% Plot wave

    # getting an example
    example_audio = train[0]["audio"]["array"]
    label = train[0]["label"]
    utils.plot_wave_and_spectogram(example_audio, label)

    # get one from the other labels as, from 0 to 9
    for label in range(0, 10):
        print(label)
        # Get the first index where the label matches
        index = next((i for i, item in enumerate(train) if item["label"] == label), None)

        if index is not None:
            example_audio = train[index]["audio"]["array"]
            utils.plot_wave_and_spectogram(example_audio, label)
    # %% get mfcc features

    fe = get_features.MFCCStatsExtractor(
        sr=8000, n_mfcc=13, drop_c0=True, use_deltas=True, use_delta_delta=True
    )
    X_tr, y_tr = fe.fit_transform(train)  # fits CMVN on train frames
    X_va, y_va = fe.transform(val)
    X_te, y_te = fe.transform(test)

    print(X_tr.shape, X_va.shape, X_te.shape)

    # %% model
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=42),
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    print("val acc:", accuracy_score(y_va, pred))
    print(classification_report(y_va, pred))

    # %%
    pred = clf.predict(X_te)
    print("val acc:", accuracy_score(y_te, pred))
    print(classification_report(y_te, pred))

    # %%
    y_pred = clf.predict(X_te)
    utils.plot_and_save_confusion_matrix(
        y_te, y_pred, train.features["label"].names, "confusion_matrix.png"
    )

    # %%

    print("-------------------------------------------------------")
    # feature latency
    t0 = time.perf_counter()
    X_te, y_te = fe.transform(test)
    t1 = time.perf_counter()
    # inference latency
    t2 = time.perf_counter()
    _ = clf.predict(X_te)
    t3 = time.perf_counter()

    feat_ms = (t1 - t0) * 1000 / len(test)
    pred_ms = (t3 - t2) * 1000 / len(test)
    # real-time factor (RTF)
    dur_s = np.sum([len(ex["audio"]["array"]) / 8000 for ex in test])
    rtf = (t1 - t0 + t3 - t2) / dur_s
    print(
        f"Feature calculation: {feat_ms:.2f} ms/utt | Prediction time: {pred_ms:.2f} ms/utt | RTF={rtf:.3f}"
    )

    # retrain model on the whole dataset
    joblib.dump({"fe": fe, "clf": clf}, "model.joblib")
    print("Model file size:", Path("model.joblib").stat().st_size / 1024, "KB")

    # %% For live testing we need the model trained with the most data possible

    print("-------------------------------------------------------")
    # Combine all datasets
    full_dataset = concatenate_datasets([train, val, test])
    print("Full dataset size:", len(full_dataset))
    print("Label distribution:", utils.label_counts(full_dataset))

    # Initialize feature extractor
    fe = get_features.MFCCStatsExtractor(
        sr=8000, n_mfcc=13, drop_c0=True, use_deltas=True, use_delta_delta=True
    )

    # Extract features from full dataset
    X_full, y_full = fe.fit_transform(full_dataset)
    print("Full feature matrix shape:", X_full.shape)

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=42),
    )
    clf.fit(X_full, y_full)

    # Save full model
    joblib.dump({"fe": fe, "clf": clf}, "full_model.joblib")
    print("Full model size:", Path("full_model.joblib").stat().st_size / 1024, "KB")
