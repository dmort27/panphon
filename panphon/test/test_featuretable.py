# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import

import unittest
import panphon.featuretable


class TestFeatureTable(unittest.TestCase):

    def setUp(self):
        self.ft = panphon.featuretable.FeatureTable()

    def test_fts_contrast2(self):
        inv = 'p t k b d ɡ a e i o u'.split(' ')
        self.assertTrue(self.ft.fts_contrast({'syl': -1}, 'voi', inv))
        self.assertFalse(self.ft.fts_contrast({'syl': 1}, 'cor', inv))
        self.assertTrue(self.ft.fts_contrast({'ant': 1, 'cor': -1}, 'voi', inv))

    def test_longest_one_seg_prefix(self):
        prefix = self.ft.longest_one_seg_prefix('pʰʲaŋ')
        self.assertEqual(prefix, 'pʰʲ')

    def test_match_pattern(self):
        self.assertTrue(self.ft.match_pattern([{'voi': -1},
                                               {'voi': 1},
                                               {'voi': -1}],
                                              'pat'))

    def test_all_segs_matching_fts(self):
        segs = self.ft.all_segs_matching_fts({'syl': -1, 'son': 1})
        self.assertIn('m', segs)
        self.assertIn('n', segs)
        self.assertIn('ŋ', segs)
        self.assertIn('m̥', segs)
        self.assertIn('l', segs)
        # self.assertNotIn('a', segs)
        # self.assertNotIn('s', segs)

    def test_word_to_vector_list_aspiration(self):
        self.assertNotEqual(self.ft.word_to_vector_list(u'pʰ'),
                            self.ft.word_to_vector_list(u'p'))

    def test_word_to_vector_list_aspiration_xsampa(self):
        self.assertNotEqual(self.ft.word_to_vector_list(u'p_h', xsampa=True),
                            self.ft.word_to_vector_list(u'p', xsampa=True))

    def test_word_to_vector_list_aspiration_xsampa_len(self):
        self.assertEqual(len(self.ft.word_to_vector_list(u'p_h', xsampa=True)), 1)

class TestIpaRe(unittest.TestCase):

    def setUp(self):
        self.ft = panphon.featuretable.FeatureTable()

    def test_compile_regex_from_str1(self):
        r = self.ft.compile_regex_from_str('[-son -cont][+syl -hi -lo]')
        self.assertIsNotNone(r.match('tʰe'))
        self.assertIsNone(r.match('pi'))

    def test_compile_regex_from_str2(self):
        r = self.ft.compile_regex_from_str('[-son -cont][+son +cont]')
        self.assertIsNotNone(r.match('pj'))
        self.assertIsNone(r.match('ts'))


class TestXSampa(unittest.TestCase):

    def setUp(self):
        self.ft = panphon.featuretable.FeatureTable()

    def test_affricates(self):
        self.assertNotEqual(self.ft.word_to_vector_list(u'tS', xsampa=True),
                            self.ft.word_to_vector_list(u't S', xsampa=True))


if __name__ == '__main__':
    unittest.main()
