import numpy as np
import pandas as pd
from typing import NamedTuple
from importlib.resources import files


class FeatureVectors(NamedTuple):
    vector_map: dict[str, np.ndarray]
    phoneme_map: dict[tuple[int], list[str]]
    feature_names: list[str]


_features = None

plus_minus_to_int = {'+': 1, '0': 0, '-': -1}


def generate_feature_vectors(feature_table='ipa_bases.csv') -> FeatureVectors:
    feature_table = files('panphon') / 'data' / feature_table
    with feature_table.open('r') as f:
        df = pd.read_csv(f)
    feature_names = df.columns[1:]
    phonemes = df['ipa']
    df[feature_names] = df[feature_names].map(lambda s: plus_minus_to_int[s])
    df[feature_names] = df[feature_names].astype(int)
    feature_vectors = np.array(df[feature_names])
    vector_map = dict(zip(phonemes, feature_vectors))

    feature_cols = df.columns[1:]  # all columns after 'ipa'

    grouped_df = df.groupby(list(feature_cols), sort=False)['ipa'].agg(';'.join).reset_index()
    grouped_df = grouped_df[['ipa'] + list(feature_cols)]

    # Now build the phoneme_map
    numeric_cols = grouped_df.columns[1:]

    phoneme_map = grouped_df.apply(
        lambda row: (tuple(row[numeric_cols]), row['ipa'].split(';')),
        axis=1
    ).to_dict()

    phoneme_map = dict(phoneme_map.values())

    return FeatureVectors(vector_map, phoneme_map, list(feature_names))
    

def get_features():
    global _features
    if _features is None:
        _features = generate_feature_vectors()
    return _features