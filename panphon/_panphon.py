# -*- coding: utf-8 -*-
from __future__ import print_function

import os.path

import pkg_resources

import regex as re
import unicodecsv as csv


# logging.basicConfig(level=logging.DEBUG)


class SegmentError(Exception):
    pass


FT_REGEX = re.compile(ur'([-+0])([a-z][A-Za-z]*)', re.U | re.X)
MT_REGEX = re.compile(ur'\[[-+0a-zA-Z ,;]*\]')
SEG_REGEX = re.compile(ur'[\p{InBasic_Latin}\p{InGreek_and_Coptic}' +
                       ur'\p{InIPA_Extensions}œ\u00C0-\u00FF]' +
                       ur'[\u0300-\u0360\u0362-\u036F]*' +
                       ur'\p{InSpacing_Modifier_Letters}*',
                       re.U | re.X)
filenames = {
    'spe+': os.path.join('data', 'ipa_all.csv'),
    'panphon': os.path.join('data', 'ipa_all.csv'),
}


def segment_text(text, seg_regex=SEG_REGEX):
    """Return an iterator of segments in the text.

    text -- string of IPA Unicode text
    seg_regex -- compiled regex defining a segment (base + modifiers)
    """
    for m in seg_regex.finditer(text):
        yield m.group(0)


def fts(s):
    """Given string with +/-[alphabetical sequence]s, return list of features.

    s -- string with +/-[alphabetical sequence]s
    """
    return [m.groups() for m in FT_REGEX.finditer(s)]


def pat(p):
    """Given a string with feature matrices (features grouped with square
    brackets into segments, return a list of sets of <value, feature> tuples.

    p - pattern as string
    """
    pattern = []
    for matrix in [m.group(0) for m in MT_REGEX.finditer(p)]:
        segment = set([m.groups() for m in FT_REGEX.finditer(matrix)])
        pattern.append(segment)
    return pattern


class FeatureTable(object):
    """Encapsulate the segment <=> feature mapping in the file
    data/segment_features.csv.

    """

    def __init__(self, feature_set='spe+'):
        filename = filenames[feature_set]
        self.segments, self.seg_dict, self.names = self._read_table(filename)
        self.seg_seq = {seg[0]: i for (i, seg) in enumerate(self.segments)}
        self.weights = self._read_weights()
        self.seg_regex = self._build_seg_regex()
        self.longest_seg = max([len(x) for x in self.seg_dict.keys()])

    def _read_table(self, filename):
        """Read the data from data/ipa_all.csv into self.segments, a
        list of 2-tuples of unicode strings and sets of feature tuples and
        self.seg_dict, a dictionary mapping from unicode segments and sets of
        feature tuples.
        """
        filename = pkg_resources.resource_filename(
            __name__, filename)
        segments = []
        with open(filename, 'rb') as f:
            reader = csv.reader(f, encoding='utf-8')
            header = reader.next()
            names = header[1:]
            for row in reader:
                seg = row[0]
                vals = row[1:]
                specs = set(zip(vals, names))
                segments.append((seg, specs))
        seg_dict = dict(segments)
        return segments, seg_dict, names

    def _read_weights(self, filename=os.path.join('data', 'feature_weights.csv')):
        filename = pkg_resources.resource_filename(
            __name__, filename)
        with open(filename, 'rb') as f:
            reader = csv.reader(f, encoding='utf-8')
            reader.next()
            weights = [float(x) for x in reader.next()]
        return weights

    def _build_seg_regex(self):
        # Build a regex that will match individual segments in a string.
        segs = sorted(self.seg_dict.keys(), key=lambda x: len(x), reverse=True)
        return re.compile(ur'({})'.format(u'|'.join(segs)))

    def delete_ties(self):
        """Deletes ties from all segments."""
        self.seg_dict = {k.replace(u'\u0361', u''): v
                         for (k, v) in self.seg_dict.items()}

    def fts(self, segment):
        """Returns features corresponding to segment as list of <value,
        feature> tuples."""
        if segment in self.seg_dict:
            return self.seg_dict[segment]
        else:
            msg = 'Segment {} is unknown.'.format(repr(segment))
            raise SegmentError(msg)

    def match(self, ft_mask, ft_seg):
        """Evaluates whether a set of features (ft_mask) are a subset of another
        set of features (ft_seg).

        ft_mask -- pattern defined as set of features (<val, name> tuples).
        ft_seg -- segment defined as a set of features (<val, name> tuples).
        """
        return set(ft_mask) <= set(ft_seg)

    def fts_match(self, features, segment):
        """Evaluates whether a set of features 'match' a segment (are a subset
        of that segment's features); returns 'None' if segment is unknown.
        """
        features = set(features)
        if segment in self.seg_dict:
            return features <= self.seg_dict[segment]
        else:
            return None

    def longest_one_seg_prefix(self, word):
        """Return longest Unicode prefix of a word."""
        for i in range(self.longest_seg, 0, -1):
            if word[:i] in self.seg_dict:
                return word[:i]
        return ''

    def segs(self, word):
        """Returns a list of segments (as strings) from a word (as a
        string).
        """
        return [m.group(1) for m in self.seg_regex.finditer(word)]

    def word_fts(self, w):
        """Returns a list of <value, feature> tuples, given a Unicode IPA
        string.

        w -- a Unicode IPA string consisting of one or more segments
        """
        return map(self.fts, self.segs(w))

    def seg_known(self, segment):
        """Returns True if segment is in segment <=> features database."""
        return segment in self.seg_dict

    def filter_string(self, s):
        """Return a string containing only legal IPA segments."""
        return ''.join(self.seg_regex.findall(s))

    def seg_fts(self, segment):
        """Returns the features of a segment as a list
        of 2-tuples, given a segment as a Unicode string; returns 'None' if segment
        is unknown.

        segment -- segment for which features are to be returned as
        Unicode string """
        if segment in self.seg_dict:
            return self.seg_dict[segment]
        else:
            return None

    def fts_intersection(self, segments):
        """Returns the features shared by all segments in the list/set of
        segments. Segments that are not known are ignored.

        segments -- set/list of features
        """
        segments = set([seg for seg
                        in segments
                        if seg in self.seg_dict])
        seg1 = segments.pop()
        fts = self.seg_dict[seg1]
        for seg in segments:
            fts = fts & self.seg_dict[seg]
        return fts

    def fts_match_any(self, fts, inv):
        """Returns a boolean based on whether there is a segment in 'inv'
        that matches all of the features in 'features'.

        features -- a collection of feature 2-tuples <val, name>
        inv -- a collection of segments represented as Unicode
               strings

        """
        return any([self.fts_match(fts, s) for s in inv])

    def fts_match_all(self, fts, inv):
        """Returns a boolean based on whether all segments in 'inv'
         matche all of the features in 'features'.

        features -- a collection of feature 2-tuples <val, name>
        inv -- a collection of segments represented as Unicode
               strings

        """
        return all([self.fts_match(fts, s) for s in inv])

    def fts_contrast2(self, fs, ft_name, inv):
        """Return True if there is a segment in inv that contrasts in feature
        ft_name.

        fs --
        ft_name -- name of the feature where contrast must be present.
        inv -- collection of segments represented as Unicode segments.
        """
        inv_fts = [self.fts(x) for x in inv if set(fs) <= self.fts(x)]
        for a in inv_fts:
            for b in inv_fts:
                if a != b:
                    diff = a ^ b
                    if len(diff) == 2:
                        if all([nm == ft_name for (_, nm) in diff]):
                            return True
        return False

    def fts_count(self, fts, inv):
        """Returns the count of segments in an inventory matching a given
        feature mask.

        fts -- feature mask given as a set of <val, name> tuples
        inv -- inventory of segments (as Unicode IPA strings)
        """
        return len(filter(lambda s: self.fts_match(fts, s), inv))

    def match_pattern(self, pat, word):
        """Implements fixed-width pattern matching. Matches just in case pattern
        is the same length (in segments) as the word and each of the segments
        in the pattern is a featural subset of the corresponding segment in the
        word. Matches return the corresponding list of feature sets; failed
        matches return None.

        pat -- pattern consisting of a sequence (list) of sets of <value,
        featured> tuples.

        word -- a Unicode IPA string consisting of zero or more segments.
        """
        segs = self.word_fts(word)
        if len(pat) != len(segs):
            return None
        else:
            if all([set(p) <= s for (p, s) in zip(pat, segs)]):
                return segs

    def match_pattern_seq(self, pat, const):
        """Implements limited pattern matching. Matches just in case pattern is
        the same length (in segments) as the constituent and each of the
        segments in the pattern is a featural subset of the corresponding
        segment in the word.

        pat -- pattern consisting of a list of sets of <value, featured>
        tuples.

        const -- a sequence of Unicode IPA strings consisting of zero or more
        segments.
        """
        segs = [self.fts(s) for s in const]
        if len(pat) != len(segs):
            return False
        else:
            return all([set(p) <= s for (p, s) in zip(pat, segs)])

    def all_segs_matching_fts(self, fts):
        """Return a segments matching a feature mask, both as <name, value>
        tuples (sorted in reverse order by length).

         fts -- feature mask as <value, name> tuples.
        """
        matching_segs = []
        for seg, pairs in self.segments:
            if set(fts) <= set(pairs):
                matching_segs.append(seg)
        return sorted(matching_segs, reverse=True)

    def compile_regex_from_str(self, ft_str):
        """Given a string describing features masks for a sequence of segments,
        return a regex matching the corresponding strings.

        ft_str -- A string consisting of feature masks, each enclosed in
        square brackets, in which the features are delimited by any
        standard delimiter. """

        sequence = []
        for m in re.finditer(ur'\[([^]]+)\]', ft_str):
            ft_mask = fts(m.group(1))
            segs = self.all_segs_matching_fts(ft_mask)
            sub_pat = u'({})'.format(u'|'.join(segs))
            sequence.append(sub_pat)
        pattern = u''.join(sequence)
        regex = re.compile(pattern)
        return regex

    def segment_to_vector(self, seg):
        """Given a Unicode IPA segment, return a list of feature specificiations
        in cannonical order.
        """
        ft_dict = {ft: val for (val, ft) in self.seg_dict[seg]}
        return [ft_dict[name] for name in self.names]

    def word_to_vector_list(self, word):
        """Return a list of feature vectors, given a Unicode IPA word.
        """
        return map(self.segment_to_vector, self.segs(word))
