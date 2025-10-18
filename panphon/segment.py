# -*- coding: utf-8 -*-


from typing import Iterator, Mapping, TypeVar, List, Dict
import regex as re

T = TypeVar('T')

class Segment(Mapping[str, int]):
    """Model a phonological segment as a vector of features.

    Constructs a `Segment` object that models a phonological segment as a 
    vector of features.

    Parameters
    ----------
    names : List[str]
        An ordered list of feature names.
    features : Dict[str, int], optional
        Name-feature pairs for specified features. Default is empty dict.
    ftstr : str, optional
        A string, each /(+|0|-)\\w+/ sequence of which is interpreted as a 
        feature specification. Default is empty string.
    weights : List[float], optional
        An ordered list of feature weights/saliences. Default is empty list,
        which results in uniform weights of 1.0.
    """
    def __init__(self, names: List[str], features: Dict[str, int] = {}, ftstr: str = '', weights: List[float] = []) -> None:
        self.n2s = {-1: '-', 0: '0', 1: '+'}
        self.s2n = {k: v for (v, k) in self.n2s.items()}
        self.names = names
        """Set a feature specification"""
        self.data = {}
        for name in names:
            if name in features:
                self.data[name] = features[name]
            else:
                self.data[name] = 0
        for m in re.finditer(r'(\+|0|-)(\w+)', ftstr):
            v, k = m.groups()
            self.data[k] = self.s2n[v]
        if weights:
            self.weights = weights
        else:
            self.weights = [1 for _ in names]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> int:
        """Get a feature specification.

        Parameters
        ----------
        key : str
            Feature name.

        Returns
        -------
        int
            Feature value (-1, 0, or 1).
        """
        return self.data[key]

    def __setitem__(self, key: str, value: int):
        """Set a feature specification.

        Parameters
        ----------
        key : str
            Feature name.
        value : int
            Feature value (-1, 0, or 1).

        Raises
        ------
        KeyError
            If the feature name is not in the list of known features.
        """
        if key in self.names:
            self.data[key] = value
        else:
            raise KeyError('Unknown feature name.')

    def __repr__(self) -> str:
        """Return a string representation of a feature vector.

        Returns
        -------
        str
            String representation showing all features and their values.
        """
        pairs = [(self.n2s[self.data[k]], k) for k in self.names]
        fts = ', '.join(['{}{}'.format(*pair) for pair in pairs])
        return '<Segment [{}]>'.format(fts)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the feature names.

        Returns
        -------
        Iterator[str]
            Iterator over feature names.
        """
        return iter(self.names)

    def items(self) -> List[tuple[str, int]]:  # type: ignore[override]
        """Return a list of the features as (name, value) pairs.

        Returns
        -------
        List[Tuple[str, int]]
            List of features as (name, value) pairs.
        """
        return [(k, self.data[k]) for k in self.names]

    def iteritems(self) -> Iterator[tuple[str, int]]:
        """Return an iterator over the features as (name, value) pairs.

        Returns
        -------
        Iterator[Tuple[str, int]]
            Iterator over features as (name, value) pairs.
        """
        return ((k, self.data[k]) for k in self.names)

    def update(self, features: Dict[str, int]) -> None:
        """Update the object's features to match `features`.

        Parameters
        ----------
        features : Dict[str, int]
            Dictionary containing the new feature values.
        """
        self.data.update(features)

    def match(self, ft_mask: 'Segment') -> bool:
        """Determine whether self's features are a superset of ft_mask's.

        Parameters
        ----------
        ft_mask : Segment
            Segment object with feature specifications to match against.

        Returns
        -------
        bool
            True if superset relationship holds, False otherwise.
        """
        return all([self.data[k] == v for (k, v) in ft_mask.items()])

    def __ge__(self, other: "Segment") -> bool:
        """Determine whether self's features are a superset of other's.

        Parameters
        ----------
        other : Segment
            Segment to compare against.

        Returns
        -------
        bool
            True if self's features are a superset of other's.
        """
        return self.match(other)

    def intersection(self, other: "Segment") -> "Segment":
        """Return Segment of features shared by self and other.

        Parameters
        ----------
        other : Segment
            Object with feature specifications.

        Returns
        -------
        Segment
            New Segment containing (name, value) pairs for each shared feature.
        """
        data = dict(set(self.items()) & set(other.items()))
        names = list(filter(lambda a: a in data, self.names))
        return Segment(names, data)

    def __and__(self, other: "Segment") -> "Segment":
        """Return Segment of features shared by `self` and `other`"""
        return self.intersection(other)

    def numeric(self, names: List[str] = []) -> List[int]:
        if not names:
            names = self.names
        """Return feature values as a list of integers"""
        return [self.data[k] for k in names]

    def strings(self, names: List[str] = []) -> List[str]:
        """Return feature values as a list of strings"""
        if not names:
            names = self.names
        return list(map(lambda x: self.n2s[x], self.numeric()))

    def distance(self, other: "Segment") -> int:
        """Compute a distance between `self` and `other`

        Args:
            other (Segment): object to compare with `self`

        Returns:
            int: the sum of the absolute value of the difference between each
                 of the feature values in `self` and `other`.
        """
        return sum(abs(a - b) for (a, b) in zip(self.numeric(), other.numeric()))

    def norm_distance(self, other: "Segment") -> float:
        """Compute a distance, normalized by vector length

        Args:
            other (Segment): object to compare with `self`

        Returns:
            float: the sum of the absolute value of the difference between
                   each of the feature values in `self` and `other`, divided
                   by the number of features per vector.
        """
        return self.distance(other) / len(self.names)

    def __sub__(self, other: "Segment") -> float:
        """Distance between segments, normalized by vector length"""
        return self.norm_distance(other)

    def hamming_distance(self, other) -> int:
        """Compute Hamming distance between feature vectors

        Args:
            other (Segment): object to compare with `self`

        Returns:
            int: the unnormalized Hamming distance between the two vectors.
        """
        return sum(int(a != b) for (a, b) in zip(self.numeric(), other.numeric()))

    def norm_hamming_distance(self, other: "Segment") -> float:
        """Compute Hamming distance, normalized by vector length

        Args:
            other (Segment): object to compare with `self`

        Returns:
            int: the normalized Hamming distance between the two vectors.
        """
        return self.hamming_distance(other) / len(self.names)

    def weighted_distance(self, other: "Segment") -> float:
        """Compute weighted distance

        Args:
            other (Segment): object to compare with `self`

        Returns:
            float: the weighted distance between the two vectors
        """
        return sum([abs(a - b) * c for (a, b, c)
                   in zip(self.numeric(), other.numeric(), self.weights)])

    def norm_weighted_distance(self, other: "Segment") -> float:
        """Compute weighted distance, normalized by vector length

        Args:
            other (Segment): object to compare with `self`

        Returns:
            float: the weighted distance between the two vectors, normalized by
                   vector length.
        """
        return self.weighted_distance(other) / sum(self.weights)

    def specified(self) -> Dict[str, int]:
        """Return dictionary of features that are specified '+' or '-' (1 or -1)

        Returns:
            dict: each feature in `self` for which the value is not 0
        """
        return {k: v for (k, v) in self.data.items() if v != 0}

    def differing_specs(self, other: "Segment") -> List[str]:
        """Return a list of feature names that differ in their specified values

        Args:
            other (Segment): object to compare with `self`

        Returns:
            list: the names of the features that differ in the two vectors
        """
        return [k for (k, v) in self.items() if other[k] != v]
