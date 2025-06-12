import re
import warnings
from collections import OrderedDict
from functools import lru_cache
from importlib.resources import files
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from yaml import safe_load

# Maximum number of iterations allowed in deriving new phonemes
MAX_STEPS = 10


# Data structure for representing mappings between feature vectors and phonemes
class FeatureVectors(NamedTuple):
    vector_map: dict[str, np.ndarray]
    phoneme_map: dict[tuple[int, ...], list[str]]
    feature_names: list[str]
    phonemes: list[str]
    feature_vectors: np.ndarray


# Data structure for representing modifiers and the corresponding transforms to
# phonemes and feature vectors
class Modifiers(NamedTuple):
    prefix_modifiers: list[str]
    postfix_modifiers: list[str]
    transforms: dict[str, tuple[np.ndarray, Callable[[str], str]]]


# Variables representing the cached data structures
_modifiers = None
_features = None
_segment_re = None

# Mapping between string and numeric representation of features
plus_minus_to_int = {'+': 1, '0': 0, '-': -1}


def vector_to_tuple(vector: np.ndarray) -> tuple:
    """
    Convert np.ndarray feature vectors to tuples of integers
    """
    return tuple(vector)


def generate_feature_vectors(feature_table='ipa_bases.csv') -> FeatureVectors:
    """
    Build a FeatureVectors object based on the contents of a feature table
    file.

    Parameters
    ----------
    feature_table : str

    Returns
    -------
    FeatureVectors
        An object with mappings from phonemes to vectors and vectors to
        phonemes
        as well as the names of the features, the phonemes themselves, and the
        feature vectors.
    """
    feature_table = files('panphon') / 'data' / feature_table
    with feature_table.open('r') as f:
        df = pd.read_csv(f)
    feature_names = df.columns[1:]
    df = df.sort_values(
        by='ipa',
        key=lambda col: col.str.len(),
        ascending=False
        )
    phonemes = df['ipa']
    df[feature_names] = df[feature_names].map(
        lambda s: plus_minus_to_int[s]
        )  # type: ignore
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

    # Ensure tuple values are plain Python ints, not np.int64
    def row_key(row):
        return tuple(int(x) for x in row[numeric_cols])

    phoneme_map_items = grouped_df.apply(
        lambda row: (
            row_key(row),
            sorted(row['ipa'].split(';'))
        ),
        axis=1
    ).tolist()

    phoneme_map = dict(phoneme_map_items)

    def handle_missing_vector(vector):
        warnings.warn(f"Vector not found as key={vector}")
        return ''

    return FeatureVectors(
        vector_map,
        phoneme_map,
        list(feature_names),
        list(phonemes),
        feature_vectors,
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


def add_and_get_new_vector(ipa: str) -> np.ndarray:
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
        tuple_vector = vector_to_tuple(vector)
        features.phoneme_map[tuple_vector] = [ipa]  # type: ignore
        return vector
    else:
        warnings.warn(f'Phoneme {ipa} cannot be analyzed.')
        return np.zeros(len(features.feature_names))


@lru_cache
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
        features.vector_map.get(m.group(0), add_and_get_new_vector(m.group(0)))
        for m in segment_re.finditer(ipa)
    ]
    return np.stack(rows)


def hamming_distance(u: np.ndarray, v: np.ndarray) -> int:
    return int(np.sum(u != v))


def add_and_get_new_phoneme(target_vector: np.ndarray) -> list[str]:
    # Obtain shared resources
    features = get_features()
    modifiers = get_modifiers()

    # Find the closest known vector to the target vector (as well as the
    # corresponding phoneme)
    distances = np.sum(features.feature_vectors != target_vector, axis=1)
    idx = np.argmin(distances)
    phoneme = features.phonemes[idx]
    vector = features.vector_map[phoneme].copy()

    # Iterate through the modifiers in multiple passes
    for _ in range(MAX_STEPS):
        found = False
        candidates = []
        # Iterate through the modifiers, trying each of them
        for _, (vector_tr, phoneme_tr) in modifiers.transforms.items():
            vector_candidate = np.where(vector_tr != 0, vector_tr, vector)
            if np.array_equal(vector_candidate, vector):
                continue
            phoneme_candidate = phoneme_tr(phoneme)
            # If a perfect match is found, stop
            if np.array_equal(vector_candidate, target_vector):
                vector = vector_candidate
                phoneme = phoneme_candidate
                found = True
                break
            # Compute the loss for each candidate
            loss = hamming_distance(vector_candidate, target_vector)
            candidates.append((loss, vector_candidate, phoneme_candidate))
        if found:
            break
        else:
            candidates = sorted(candidates, key=lambda x: x[0])
            best_loss, best_vector, best_phoneme = candidates.pop(0)
            if best_loss >= hamming_distance(target_vector, vector):
                break
            else:
                vector, phoneme = best_vector, best_phoneme
    features.phoneme_map[vector_to_tuple(target_vector)] = [phoneme]
    return [phoneme]


def decode(matrix: np.ndarray) -> str:
    """
    Decode a feature matrix into an IPA string.

    Parameters
    ----------
    matrix : np.ndarray
        A matrix encoding a sequence of phonemes.

    Returns
    -------
    str
        A string of phonemes corresponding the the input feature matrix.
    """

    features = get_features()

    def get_phoneme(vector):
        vector = vector_to_tuple(vector)
        if vector in features.phoneme_map:
            phonemes = features.phoneme_map[vector]
            return phonemes[0]
        else:
            return add_and_get_new_phoneme(vector)[0]  # type: ignore

    return ''.join([get_phoneme(row) for row in matrix])
