from __future__ import division, print_function, unicode_literals

import copy
import os
import codecs

import pkg_resources
import yaml

import _panphon
import regex as re
import unicodecsv as csv


def flip(s):
    return map(lambda (a, b): (b, a), s)


def update_ft_set(seg, dia):
    seg = dict(flip(seg))
    seg.update(dia)
    return flip(set(seg.items()))


class PermissiveFeatureTable(_panphon.FeatureTable):
    """Encapsulate the segment <=> feature vector mapping implied by the files
    data/ipa_all.csv and diacritic_definitions.yml"""

    def __init__(self,
                 ipa_bases=os.path.join('data', 'ipa_bases.csv'),
                 dias=os.path.join('data', 'diacritic_definitions.yml'),
                 ):
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
        seg_re = '(?P<pre>({})*)(?P<base>{})(?P<post>({})*)'.format(pre_jnd, bases_jnd, post_jnd)
        return re.compile(pre_re), re.compile(post_re), re.compile(seg_re)

    def fts(self, segment):
        match = self.seg_regex.match(segment)
        if match:
            pre, base, post = match.group('pre'), match.group('base'), match.group('post')
            seg = copy.deepcopy(self.bases[base])
            for m in reversed(pre):
                seg = update_ft_set(seg, self.prefix_dias[m])
            for m in post:
                seg = update_ft_set(seg, self.postfix_dias[m])
            return seg
        else:
            raise _panphon.SegmentError
