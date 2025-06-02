import unittest
from panphon import panphonvec
import numpy as np


class TestFeature(unittest.TestCase):
                  
    def setUp(self):
        self.feature_vectors = panphonvec.get_features()

    def test_feature_vectors(self):
        self.assertEqual(len(self.feature_vectors.vector_map['a']), 24)

    def test_phonemes1(self):
        self.assertFalse(
            np.array_equal(
                self.feature_vectors.vector_map['t'],
                self.feature_vectors.vector_map['p']
            )
        )

    def test_round_trip(self):
        vec = self.feature_vectors.vector_map['k']
        vec_tuple = tuple(vec)
        self.assertEqual(self.feature_vectors.phoneme_map[vec_tuple], ['k'])

    