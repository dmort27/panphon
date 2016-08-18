# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import unittest
import _panphon


class TestFeatureTable(unittest.TestCase):

    def setUp(self):
        self.ft = _panphon.FeatureTable()

    def test_fts_contrast2(self):
        inv = 'p t k b d ɡ a e i o u'.split(' ')
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


if __name__ == '__main__':
    unittest.main()
