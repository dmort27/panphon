from __future__ import print_function, absolute_import, unicode_literals

import codecs
import copy
import os

import pkg_resources
import yaml

import regex as re
import unicodecsv as csv
from . import _panphon


def flip(s):
    return [(b, a) for (a, b) in s]


def update_ft_set(seg, dia):
    seg = dict(flip(seg))
    seg.update(dia)
    return flip(set(seg.items()))


class PermissiveFeatureTable(_panphon.FeatureTable):
    """Encapsulate the segment <=> feature vector mapping implied by the files
    data/ipa_all.csv and diacritic_definitions.yml. Uses a more permissive
    algorithm for identifying base+diacritic combinations. To avoid a
    combinatorial explosion, it never generates all of the dia^a+base+base^b
    combinations, meaning it cannot make statements about the whole set of
    segments."""

    def __init__(self,
                 feature_set='spe+',
                 feature_model='strict',
                 ipa_bases=os.path.join('data', 'ipa_bases.csv'),
                 dias=os.path.join('data', 'diacritic_definitions.yml'),
                 ):
        dias = pkg_resources.resource_filename(__name__, dias)
        self.bases, self.names = self._read_ipa_bases(ipa_bases)
        self.prefix_dias, self.postfix_dias = self._read_dias(dias)
        self.pre_regex, self.post_regex, self.seg_regex = self._compile_seg_regexes(self.bases, self.prefix_dias, self.postfix_dias)

    def _read_ipa_bases(self, fn):
        fn = pkg_resources.resource_filename(__name__, fn)
        with open(fn, 'rb') as f:
            reader = csv.reader(f, encoding='utf-8', delimiter=str(','))
            names = next(reader)[1:]
            bases = {}
            for row in reader:
                seg, vals = row[0], row[1:]
                bases[seg] = (set(zip(vals, names)))
        return bases, names

    def _read_dias(self, fn):
        prefix, postfix = {}, {}
        with codecs.open(fn, 'r', 'utf-8') as f:
            defs = yaml.load(f.read())
            for dia in defs['diacritics']:
                if dia['position'] == 'pre':
                    prefix[dia['marker']] = dia['content']
                else:
                    postfix[dia['marker']] = dia['content']
        return prefix, postfix

    def _compile_seg_regexes(self, bases, prefix, postfix):
        pre_jnd = '|'.join(prefix.keys())
        post_jnd = '|'.join(postfix.keys())
        bases_jnd = '|'.join(bases.keys())
        pre_re = '({})'.format(pre_jnd)
        post_re = '({})'.format(post_jnd)
        seg_re = '(?P<all>(?P<pre>({})*)(?P<base>{})(?P<post>({})*))'.format(pre_jnd, bases_jnd, post_jnd)
        return re.compile(pre_re), re.compile(post_re), re.compile(seg_re)

    def fts(self, segment):
        """Returns features corresponding to segment as list of <value,
        feature> tuples.

        segment -- segment for which features are to be returned as
        Unicode string.

        Return None if the segment is unknown"""
        match = self.seg_regex.match(segment)
        if match:
            pre, base, post = match.group('pre'), match.group('base'), match.group('post')
            seg = copy.deepcopy(self.bases[base])
            for m in reversed(pre):
                seg = update_ft_set(seg, self.prefix_dias[m])
            for m in post:
                seg = update_ft_set(seg, self.postfix_dias[m])
            return set(seg)
        else:
            return None

    def fts_match(self, features, segment):
        """Evaluates whether a set of features 'match' a segment (are a subset
        of that segment's features); returns 'None' if segment is unknown.
        """
        features = set(features)
        fts = self.fts(segment)
        if fts:
            return features <= fts
        else:
            return False

    def longest_one_seg_prefix(self, word):
        """Return longest IPA Unicode prefix of a word."""
        match = self.seg_regex.match(word)
        if match:
            return match.group(0)
        else:
            return ''

    def seg_known(self, segment):
        """Return True if the segment is valid given the known set of bases and
        diacritics.

        segment -- a string which may or may not be a valid segment
        """
        if self.seg_regex.match(segment):
            return True
        else:
            return False

    def filter_segs(self, segs):
        """Given list of strings, return only those which are valid segments."""
        def whole_seg(seg):
            m = self.seg_regex.match(seg)
            if m and m.group(0) == seg:
                return True
            else:
                return False
        return list(filter(whole_seg, segs))

    @property
    def all_segs_matching_fts(self):
        raise AttributeError("'PermissiveFeatureTable' object has no attribute 'all_segs_matching_fts'")
