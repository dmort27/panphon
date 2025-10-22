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
    phonemes: list[str]
    feature_matrix: np.ndarray
    phoneme_to_index: dict[str, int]
    vector_to_index: dict[bytes, int]
    feature_names: list[str]


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
plus_minus_to_int = {"+": 1, "0": 0, "-": -1}


def vector_to_hash(vector: np.ndarray) -> bytes:
    """
    Convert np.ndarray feature vector to hashable bytes representation.
    """
    return vector.tobytes()


def generate_feature_vectors(feature_table="ipa_bases.csv") -> FeatureVectors:
    """
    Build a FeatureVectors object based on the contents of a feature table
    file.

    Parameters
    ----------
    feature_table : str

    Returns
    -------
    FeatureVectors
        An object with a list of phonemes, a feature matrix, and mappings
        from phonemes to row indices and from vectors to row indices.
    """
    feature_table = files("panphon") / "data" / feature_table
    with feature_table.open("r", encoding="utf-8") as f:
        df = pd.read_csv(f)
    feature_names: list[str] = list(df.columns[1:].astype(str))

    # Sort by length descending for proper regex matching (longest first)
    df = df.sort_values(by="ipa", key=lambda col: col.str.len(), ascending=False)

    phonemes: list[str] = list(df["ipa"].astype(str))

    # Convert feature values to integers
    df[feature_names] = df[feature_names].map(lambda s: plus_minus_to_int[s])  # type: ignore
    df[feature_names] = df[feature_names].astype(int)

    # Create the feature matrix (single ndarray)
    feature_matrix = np.array(df[feature_names], dtype=np.int8)

    # Build phoneme_to_index mapping
    phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(phonemes)}

    # Build vector_to_index mapping (hash of vector -> row index)
    vector_to_index = {}
    for idx, vector in enumerate(feature_matrix):
        vector_hash = vector_to_hash(vector)
        # For duplicate vectors, we keep only the first occurrence
        if vector_hash not in vector_to_index:
            vector_to_index[vector_hash] = idx

    return FeatureVectors(
        phonemes,
        feature_matrix,
        phoneme_to_index,
        vector_to_index,
        list(feature_names),
    )


def get_features():
    global _features
    if _features is None:
        _features = generate_feature_vectors()
    return _features


def generate_modifiers(definitions_fn: str = "diacritic_definitions.yml") -> Modifiers:
    features = get_features()

    def compute_mod_vector(content: dict[str, str]) -> np.ndarray:
        vector = np.zeros(len(features.feature_names))
        for name, value in content.items():
            idx = features.feature_names.index(name)
            numeric_value = plus_minus_to_int[value]
            vector[idx] = numeric_value
        return vector

    with (files("panphon") / "data" / definitions_fn).open(encoding="utf-8") as f:
        definitions = safe_load(f)
    prefix = []
    postfix = []
    transforms = OrderedDict()
    for modifier in definitions["diacritics"]:
        marker = modifier["marker"]
        vector = compute_mod_vector(modifier["content"])
        if modifier["position"] == "pre":
            prefix.append(marker)
            transforms[marker] = (vector, (lambda m: lambda x: m + x)(marker))
        else:
            postfix.append(marker)
            transforms[marker] = (vector, (lambda m: lambda x: x + m)(marker))
    return Modifiers(prefix, postfix, transforms)


def get_modifiers():
    global _modifiers
    if _modifiers is None:
        _modifiers = generate_modifiers()
    return _modifiers


def build_segment_re() -> re.Pattern[str]:
    feature_vectors = get_features()
    modifiers = get_modifiers()
    segment_re = re.compile(
        f"""
        ([{"".join(modifiers.prefix_modifiers)}]*)
        ({"|".join(feature_vectors.phonemes)})
        ([{"".join(modifiers.postfix_modifiers)}]*)
        """,
        re.X,
    )
    return segment_re


def get_segment_re() -> re.Pattern:
    global _segment_re
    if _segment_re is None:
        _segment_re = build_segment_re()
    return _segment_re


def compute_phoneme_vector(ipa: str) -> tuple[np.ndarray, str] | None:
    """
    Compute the feature vector for a phoneme that may include diacritics.

    Returns tuple of (vector, canonical_form) or None if phoneme cannot be analyzed.
    """
    features = get_features()
    modifiers = get_modifiers()
    segment_re = get_segment_re()

    # Check whether the input string matches the regular expression for segments
    if match := segment_re.match(ipa):
        pre, base, post = match.groups()

        # Get base phoneme index and copy its vector
        base_idx = features.phoneme_to_index[base]
        vector = features.feature_matrix[base_idx].copy()

        # Iterate through the modifiers, updating the feature representations
        for marker in post + pre:
            feature_tr, _ = modifiers.transforms[marker]
            vector[feature_tr != 0] = feature_tr[feature_tr != 0]

        return vector, ipa
    else:
        return None


@lru_cache(maxsize=10000)
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
        value 1 indicates an active feature (+), the value -1 indicates an
        inactive feature (-), and the value 0 indicates an irrelevant feature.
    """
    segment_re = get_segment_re()
    features = get_features()

    rows = []
    for m in segment_re.finditer(ipa):
        phoneme = m.group(0)

        # Try to get from phoneme_to_index first (for base phonemes)
        if phoneme in features.phoneme_to_index:
            idx = features.phoneme_to_index[phoneme]
            rows.append(features.feature_matrix[idx])
        else:
            # Compute vector for phoneme with diacritics
            result = compute_phoneme_vector(phoneme)
            if result is not None:
                vector, _ = result
                rows.append(vector)
            else:
                warnings.warn(f"Phoneme {phoneme} cannot be analyzed.")
                rows.append(np.zeros(len(features.feature_names), dtype=np.int8))

    return (
        np.stack(rows) if rows else np.array([]).reshape(0, len(features.feature_names))
    )


def hamming_distance(u: np.ndarray, v: np.ndarray) -> int:
    return int(np.sum(u != v))


def find_closest_phoneme(target_vector: np.ndarray) -> str:
    """
    Find the closest phoneme to the target vector by applying diacritics.
    """
    features = get_features()
    modifiers = get_modifiers()

    # Find the closest known vector to the target vector (as well as the
    # corresponding phoneme)
    distances = np.sum(features.feature_matrix != target_vector, axis=1)
    idx = int(np.argmin(distances))
    phoneme = features.phonemes[idx]
    vector = features.feature_matrix[idx].copy()

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
            if not candidates:
                break
            candidates = sorted(candidates, key=lambda x: x[0])
            best_loss, best_vector, best_phoneme = candidates.pop(0)
            if best_loss >= hamming_distance(target_vector, vector):
                break
            else:
                vector, phoneme = best_vector, best_phoneme

    return phoneme


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

    phonemes = []
    for vector in matrix:
        # Try exact match first
        vector_hash = vector_to_hash(vector)
        if vector_hash in features.vector_to_index:
            idx = features.vector_to_index[vector_hash]
            phonemes.append(features.phonemes[idx])
        else:
            # Find closest phoneme using diacritics
            phoneme = find_closest_phoneme(vector)
            phonemes.append(phoneme)

    return "".join(phonemes)


# Legacy compatibility - maintain old interface
def get_vector_for_phoneme(phoneme: str) -> np.ndarray:
    """Get feature vector for a phoneme (including those with diacritics)."""
    features = get_features()
    if phoneme in features.phoneme_to_index:
        idx = features.phoneme_to_index[phoneme]
        return features.feature_matrix[idx].copy()
    else:
        result = compute_phoneme_vector(phoneme)
        if result is not None:
            return result[0]
        else:
            warnings.warn(f"Phoneme {phoneme} not found.")
            return np.zeros(len(features.feature_names), dtype=np.int8)
