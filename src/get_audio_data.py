#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 12:25:15 2025

@author: gabriel
"""


from datasets import load_dataset, Audio


def get_hf_audio_data():

    ds = load_dataset("mteb/free-spoken-digit-dataset")

    ds = ds.cast_column("audio", Audio(sampling_rate=8000))

    # Make a small val split from train (stratified)
    splits = ds["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
    train, val, test = splits["train"], splits["test"], ds["test"]

    return train, val, test
