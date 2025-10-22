import unittest
from panphon import panphonvec
import numpy as np


class TestFeature(unittest.TestCase):
    def setUp(self):
        self.feature_vectors = panphonvec.get_features()

    def test_feature_vectors(self):
        # Test that phoneme 'a' has 24 features
        idx = self.feature_vectors.phoneme_to_index["a"]
        self.assertEqual(len(self.feature_vectors.feature_matrix[idx]), 24)

    def test_phonemes1(self):
        # Test that 't' and 'p' have different feature vectors
        t_idx = self.feature_vectors.phoneme_to_index["t"]
        p_idx = self.feature_vectors.phoneme_to_index["p"]
        self.assertFalse(
            np.array_equal(
                self.feature_vectors.feature_matrix[t_idx],
                self.feature_vectors.feature_matrix[p_idx],
            )
        )

    def test_phoneme_map(self):
        # Test that we can find phoneme 'c' by its feature vector
        vector = np.array(
            [
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                0,
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                0,
                -1,
                0,
                0,
            ],
            dtype=np.int8,
        )
        vector_hash = panphonvec.vector_to_hash(vector)
        self.assertIn(vector_hash, self.feature_vectors.vector_to_index)
        idx = self.feature_vectors.vector_to_index[vector_hash]
        self.assertEqual(self.feature_vectors.phonemes[idx], "c")

    def test_vector_map(self):
        # Test that phoneme 'c' maps to the correct feature vector
        vector = np.array(
            [
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                0,
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                0,
                -1,
                0,
                0,
            ],
            dtype=np.int8,
        )
        c_idx = self.feature_vectors.phoneme_to_index["c"]
        new_vector = self.feature_vectors.feature_matrix[c_idx]
        self.assertTrue(np.array_equal(vector, new_vector))

    def test_round_trip(self):
        # Test that we can go from phoneme -> vector -> phoneme
        k_idx = self.feature_vectors.phoneme_to_index["k"]
        vector = self.feature_vectors.feature_matrix[k_idx]
        vector_hash = panphonvec.vector_to_hash(vector)
        idx = self.feature_vectors.vector_to_index[vector_hash]
        self.assertEqual(self.feature_vectors.phonemes[idx], "k")


class TestEncodeDecode(unittest.TestCase):
    def setUp(self):
        self.encode = panphonvec.encode
        self.decode = panphonvec.decode

    def test_round_trip3(self):
        # Note: 'o' and 'ɞ' have identical feature vectors, so decoding may
        # return either one. This test accepts both as valid.
        result = self.decode(self.encode("ox"))
        self.assertIn(result, ["ox", "ɞx"])

    def test_round_trip1(self):
        self.assertEqual(self.decode(self.encode("wɛlp")), "wɛlp")

    def test_round_trip2(self):
        self.assertEqual(self.decode(self.encode("pʰʲa")), "pʰʲa")

    def test_decode1(self):
        v = np.array(
            [
                [
                    -1,
                    -1,
                    1,
                    -1,
                    -1,
                    -1,
                    -1,
                    0,
                    -1,
                    1,
                    -1,
                    1,
                    -1,
                    0,
                    1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    0,
                    -1,
                    0,
                    0,
                ]
            ],
            dtype=np.int8,
        )
        self.assertEqual(self.decode(v), "pʰ")
