# -*- coding: utf-8 -*-

import unittest
import panphon._panphon as panphon

dim = 24


class TestFeatureTableAPI(unittest.TestCase):

    def setUp(self):
        self.ft = panphon.FeatureTable()

    def test_fts(self):
        self.assertEqual(len(self.ft.fts('u')), 24)

    # def test_seg_fts(self):
    #     self.assertEqual(len(self.ft.seg_fts('p')), 24)

    def test_match(self):
        self.assertTrue(self.ft.match(self.ft.fts('u'), self.ft.fts('u')))

    def test_fts_match(self):
        self.assertTrue(self.ft.fts_match(self.ft.fts('u'), 'u'))

    def test_longest_one_seg_prefix(self):
        self.assertEqual(self.ft.longest_one_seg_prefix('pap'), 'p')

    def test_validate_word(self):
        self.assertTrue(self.ft.validate_word('tik'))

    def test_segs(self):
        self.assertEqual(self.ft.segs('tik'), ['t', 'i', 'k'])

    def test_word_fts(self):
        self.assertEqual(len(self.ft.word_fts('tik')), 3)

    def test_seg_known(self):
        self.assertTrue(self.ft.seg_known('t'))

    def test_filter_string(self):
        self.assertEqual(len(self.ft.filter_string('pup$')), 3)

    def test_segs_safe(self):
        self.assertEqual(len(self.ft.segs_safe('pup$')), 4)

    def test_filter_segs(self):
        self.assertEqual(len(self.ft.filter_segs(['p', 'u', 'p', '$'])), 3)

    def test_fts_intersection(self):
        self.assertIn(('-', 'voi'), self.ft.fts_intersection(['p', 't', 'k']))

    def test_fts_match_any(self):
        self.assertTrue(self.ft.fts_match_any([('-', 'voi')], ['p', 'o', '$']))

    def test_fts_match_all(self):
        self.assertTrue(self.ft.fts_match_all([('-', 'voi')], ['p', 't', 'k']))

    def test_fts_contrast2(self):
        self.assertTrue(self.ft.fts_contrast2([], 'voi', ['p', 'b', 'r']))

    def test_fts_count(self):
        self.assertEqual(self.ft.fts_count([('-', 'voi')], ['p', 't', 'k', 'r']), 3)
        self.assertEqual(self.ft.fts_count([('-', 'voi')], ['r', '$']), 0)

    def test_match_pattern(self):
        self.assertEqual(len(self.ft.match_pattern([set([('-', 'voi')])], 'p')), 1)

    def test_match_pattern_seq(self):
        self.assertTrue(self.ft.match_pattern_seq([set([('-', 'voi')])], 'p'))

    def test_all_segs_matching_fts(self):
        self.assertIn('p', self.ft.all_segs_matching_fts([('-', 'voi')]))

    def test_compile_regex_from_str(self):
        pass

    def test_segment_to_vector(self):
        self.assertEqual(len(self.ft.segment_to_vector('p')), 24)

    def test_word_to_vector_list(self):
        self.assertEqual(len(self.ft.word_to_vector_list('pup')), 3)
