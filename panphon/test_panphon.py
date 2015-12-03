# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import unittest
import _panphon


class TestFeatureTable(unittest.TestCase):

    def setUp(self):
        self.ft = _panphon.FeatureTable()

    def test_fts_contrast2(self):
        inv = 'p t k b d g a e i o u'.split(' ')
        self.assertTrue(self.ft.fts_contrast2([('-', 'syl')], 'voi', inv))
        self.assertFalse(self.ft.fts_contrast2([('+', 'syl')], 'cor', inv))
        self.assertTrue(self.ft.fts_contrast2(_panphon.fts('+ant -cor'), 'voi', inv))


class TestIpaRe(unittest.TestCase):

    def setUp(self):
        self.ft = _panphon.FeatureTable()

    def test_compile_regex_from_str1(self):
        r = self.ft.compile_regex_from_str('[-son -cont][+syl -hi -lo]')
        self.assertIsNotNone(r.match('tʰe'))
        self.assertIsNone(r.match('pi'))

    def test_compile_regex_from_str2(self):
        r = self.ft.compile_regex_from_str('[-son -cont][+son +cont]')
        self.assertIsNotNone(r.match('pj'))
        self.assertIsNone(r.match('ts'))


class TestSonority(unittest.TestCase):

    def setUp(self):
        self.ft = _panphon.FeatureTable()

    def test_sonority_nine(self):
        segs = ['a', 'ɑ', 'æ', 'ɒ', 'e', 'o']
        scores = [9] * 6
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_eight(self):
        segs = ['i', '', 'ɨ', 'ʉ', 'ɯ', '']
        scores = [8] * 6
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_seven(self):
        segs = ['j', 'w', 'ʋ', 'ɰ', 'ɹ']
        scores = [7] * 5
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_six(self):
        segs = ['l', 'ɭ', 'r', 'ɾ']
        scores = [6] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_five(self):
        segs = ['n', 'm', 'ŋ', 'ɴ']
        scores = [5] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_four(self):
        segs = ['v', 'z', 'ʒ', 'ɣ']
        scores = [4] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_three(self):
        segs = ['f', 's', 'x', 'ħ', 'ʃ']
        scores = [3] * 5
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_two(self):
        segs = ['b', 'g', 'd', 'ɢ']
        scores = [2] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_one(self):
        segs = ['p', 'k', 'c', 'q']
        scores = [1] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

if __name__ == '__main__':
    unittest.main()
