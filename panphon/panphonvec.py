import re
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from yaml import safe_load
from typing import NamedTuple, Callable
from importlib.resources import files


class FeatureVectors(NamedTuple):
    vector_map: dict[str, np.ndarray]
    phoneme_map: dict[tuple[int], list[str]]
    feature_names: list[str]
    phonemes: list[str]


class Modifiers(NamedTuple):
    prefix_modifiers: list[str]
    postfix_modifiers: list[str]
    transforms: dict[str, tuple[np.ndarray, Callable[[str], str]]]


DIM = 24

_modifiers = None
_features = None
_segment_re = None

plus_minus_to_int = {'+': 1, '0': 0, '-': -1}


def generate_feature_vectors(feature_table='ipa_bases.csv') -> FeatureVectors:
    feature_table = files('panphon') / 'data' / feature_table
    with feature_table.open('r') as f:
        df = pd.read_csv(f)
    feature_names = df.columns[1:]
    phonemes = sorted(df['ipa'], key=len, reverse=True)
    df[feature_names] = df[feature_names].map(lambda s: plus_minus_to_int[s])
    df[feature_names] = df[feature_names].astype(int)
    feature_vectors = np.array(df[feature_names])
    vector_map = dict(zip(phonemes, feature_vectors))

    feature_cols = df.columns[1:]  # all columns after 'ipa'

    grouped_df = df.groupby(
        list(feature_cols), sort=False
    )['ipa'].agg(';'.join).reset_index()
    grouped_df = grouped_df[['ipa'] + list(feature_cols)]

    # Now build the phoneme_map
    numeric_cols = grouped_df.columns[1:]

    phoneme_map = grouped_df.apply(
        lambda row: (tuple(row[numeric_cols]), row['ipa'].split(';')),
        axis=1
    ).to_dict()

    phoneme_map = dict(phoneme_map.values())

    return FeatureVectors(
        vector_map, phoneme_map, list(feature_names), list(phonemes)
    )


def get_features():
    global _features
    if _features is None:
        _features = generate_feature_vectors()
    return _features


def generate_modifiers(
        definitions_fn: str =
        'diacritic_definitions.yml'
        ) -> Modifiers:
    features = get_features()

    def compute_mod_vector(content: dict[str, str]) -> np.ndarray:
        vector = np.zeros(len(features.feature_names))
        for name, value in content.items():
            idx = features.feature_names.index(name)
            numeric_value = plus_minus_to_int[value]
            vector[idx] = numeric_value
        return vector

    with (files('panphon') / 'data' / definitions_fn).open() as f:
        definitions = safe_load(f)
    prefix = []
    postfix = []
    transforms = OrderedDict()
    for modifier in definitions['diacritics']:
        marker = modifier['marker']
        vector = compute_mod_vector(modifier['content'])
        if modifier['position'] == 'pre':
            prefix.append(marker)
            transforms[marker] = (vector, lambda x, m=marker: m + x)
        else:
            postfix.append(marker)
            transforms[marker] = (vector, lambda x, m=marker: x + m)
    return Modifiers(prefix, postfix, transforms)


def get_modifiers():
    global _modifiers
    if _modifiers is None:
        _modifiers = generate_modifiers()
    return _modifiers


def build_segment_re() -> re.Pattern:
    feature_vectors = get_features()
    modifiers = get_modifiers()
    segment_re = re.compile(
        f"""
        ([{''.join(modifiers.prefix_modifiers)}]*)
        ({'|'.join(feature_vectors.phonemes)})
        ([{''.join(modifiers.postfix_modifiers)}]*)
        """, re.X)
    return segment_re


def get_segment_re() -> re.Pattern:
    global _segment_re
    if _segment_re is None:
        _segment_re = build_segment_re()
    return _segment_re


def get_new_vector(ipa: str) -> np.ndarray:
    # Access the shared data structure
    features = get_features()
    modifiers = get_modifiers()
    segment_re = get_segment_re()

    # Check whether the input string matches the regular expression for
    # segments
    if match := segment_re.match(ipa):
        pre, base, post = match.groups()
        vector = features.vector_map[base].copy()

        # Iterate through the modifiers, updating the feature representations
        for marker in (post + pre):
            feature_tr, _ = modifiers.transforms[marker]
            vector[feature_tr != 0] = feature_tr[feature_tr != 0]
        features.vector_map[ipa] = vector
        features.phoneme_map[tuple(vector)] = [ipa]
        return vector
    else:
        warnings.warn(f'Phoneme {ipa} cannot be analyzed.')
        return np.zeros(DIM)


def encode(ipa: str) -> np.ndarray:
    """
    Encode an IPA string as a NumPy array representing the features of each
    segment.

    Parameters
    ----------
    ipa : str
       The string of phonemes, represented in IPA, to be converted to vectors.

    Returns
    -------
    np.ndarray
        An array of integers in which each row corresponds to a phoneme. The
        value 1 indicates an active feature (+), the value -1 indicates and
        inactive feature (-), and the value 0 indicates an irrelevant feature.
    """
    segment_re = get_segment_re()
    features = get_features()
    rows = [
        features.vector_map.get(m.group(0), get_new_vector(m.group(0)))
        for m in segment_re.finditer(ipa)
    ]
    return np.stack(rows)
