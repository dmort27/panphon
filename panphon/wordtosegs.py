# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, unicode_literals

import logging
import unicodedata

import epitran
import epitran.space
import panphon


class WordToSegs(object):
    def __init__(self, code, space_name):
        self.epi = epitran.Epitran(code)
        self.ft = panphon.FeatureTable()
        self.space = epitran.space.Space(space_name)
        self.num_panphon_fts = len(self.ft.names)

    def word_to_segs(self, word, normpunc=False):
        """Returns feature vectors, etc. for segments and punctuation in a word.

        word -- Unicode string representing a word in the orthography specified
                when the class is instantiated.
        return -- a list of tuples, each representing an IPA segment or a
                  punctuation character. Tuples consist of <category, lettercase,
                  orthographic_form, phonetic_form, id, feature_vector>.

        Category consists of the standard Unicode classes (e.g. 'L' for letter
        and 'P' for punctuation). Case is binary: 1 for uppercase and 0 for
        lowercase.
        """

        def cat_and_cap(c):
            cat, case = tuple(unicodedata.category(c))
            case = 1 if case == 'u' else 0
            return unicode(cat), case

        def recode_ft(ft):
            if ft == '+':
                return 1
            elif ft == '0':
                return 0
            elif ft == '-':
                return -1

        def vec2bin(vec):
            return map(recode_ft, vec)

        def to_vector(seg):
            if seg == '':
                return [0] * self.num_panphon_fts
            else:
                return vec2bin(self.ft.segment_to_vector(seg))

        def to_space(seg):
            if seg in self.space.dict:
                return self.space[seg]
            else:
                return -1

        tuples = []
        _, capitalized = cat_and_cap(word[0])
        first = True
        trans = self.epi.transliterate(word, normpunc)
        while trans:
            match = self.ft.seg_regex.match(trans)
            if match:
                span = match.group(1)
                case = capitalized if first else 0
                first = False
                logging.debug(u'span = "{}" (letter)'.format(span))
                tuples.append(('L', case, span, span, to_space(span), to_vector(span)))
                trans = trans[len(span):]
                logging.debug(u'trans = "{}" (letter)'.format(trans))
            else:
                span = trans[0]
                logging.debug('span = "{}" (non-letter)'.format(span))
                span = self.normalize_punc(span) if normpunc else span
                cat, case = cat_and_cap(span)
                cat = 'P' if normpunc and cat in self.puncnorm.puncnorm else cat
                tuples.append((cat, case, span, '', to_space(span), to_vector('')))
                trans = trans[1:]
                logging.debug(u'trans = "{}" (non-letter)'.format(trans))
        return tuples
