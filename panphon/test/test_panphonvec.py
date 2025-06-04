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

    def test_phoneme_map(self):
        vector = np.array([-1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 1, -1, -1, -1, -1, 0, -1, 0, 0])
        # tuple_vector = tuple(int(x) for x in vector)
        self.assertIn(
            "c", self.feature_vectors.phoneme_map[panphonvec.vector_to_tuple(vector)]
        )  # type: ignore

    def test_vector_map(self):
        vector = np.array([-1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 1, -1, -1, -1, -1, 0, -1, 0, 0])
        new_vector = self.feature_vectors.vector_map['c']
        self.assertTrue(
            np.array_equal(
                vector,
                new_vector
            )
        )

    def test_round_trip(self):
        vector = self.feature_vectors.vector_map['k']
        self.assertEqual(
            self.feature_vectors.phoneme_map[panphonvec.vector_to_tuple(vector)], ['k']
        )  # type: ignore


class TestEncodeDecode(unittest.TestCase):

    def setUp(self):
        self.encode = panphonvec.encode
        self.decode = panphonvec.decode

    def test_round_trip3(self):
        self.assertEquals(self.decode(self.encode('ox')), 'ox')

    def test_round_trip1(self):
        self.assertEquals(self.decode(self.encode('wɛlp')), 'wɛlp')

    def test_round_trip2(self):
        self.assertEquals(self.decode(self.encode('pʰʲa')), 'pʰʲa')

    def test_decode1(self):
        v = np.array([[-1, -1,  1, -1, -1, -1, -1,  0, -1,  1, -1,  1, -1,  0,  1, -1, -1, -1, -1, -1,  0, -1,  0,  0]])
        self.assertEqual(self.decode(v), 'pʰ')
