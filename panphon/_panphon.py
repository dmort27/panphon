# -*- coding: utf-8 -*-
from __future__ import print_function

import pkg_resources
import regex as re
import unicodecsv as csv
import copy


class FeatureError(Exception):
    pass


class SegmentError(Exception):
    pass


FT_REGEX = re.compile(ur'([-+0])([a-z][A-Za-z]*)', re.U | re.X)


def fts(s):
    return [m.groups() for m in FT_REGEX.finditer(s)]

filenames = {
    'spe+': 'data/segment_features.csv',
    'panphon': 'data/segment_features.csv',
    'phoible': 'data/segment_features_phoible.csv',
    }


class FeatureTable(object):
    """Encapsulate the segment <=> feature mapping in the file
    data/segment_features.csv.

    """
    def __init__(self, feature_set='spe+'):
        filename = filenames[feature_set]
        self._read_table(filename)

    def _read_table(self, filename):
        """Read the data from data/segment_features.csv into self.segments, a
        list of 2-tuples of unicode strings and sets of feature tuples
        and self.seg_dict, a dictionary mapping from unicode segments
        and sets of feature tuples.

        """
        filename = pkg_resources.resource_filename(
            __name__, filename)
        self.segments = []
        with open(filename, 'rb') as f:
            reader = csv.reader(f, encoding='utf-8')
            header = reader.next()
            names = header[1:]
            for row in reader:
                seg = row[0]
                vals = row[1:]
                specs = set(zip(vals, names))
                self.segments.append((seg, specs))
        self.seg_dict = dict(self.segments)
        self.names = names

    def delete_ties(self):
        """Deletes ties from all segments."""
        self.seg_dict = {k.replace(u'\u0361', u''): v
                         for (k, v) in self.seg_dict.items()}

    def fts(self, segment):
        """Returns features corresponding to segment as list of <feature,
        value> tuples."""
        if segment in self.seg_dict:
            return self.seg_dict[segment]
        else:
            return None

    def fts_match(self, features, segment):
        """Evaluates whether a set of features 'match' a segment (are a subset
        of that segment's features); returns 'None' if segment is unknown.
        """
        features = set(features)
        if segment in self.seg_dict:
            return features <= self.seg_dict[segment]
        else:
            return None

    def seg_known(self, segment):
        """Returns True if segment is in segment <=> features database."""
        return segment in self.seg_dict

    def seg_fts(self, segment):
        """Returns the features as a list of 2-tuples, given a segment as a
        Unicode string; returns 'None' if segment is unknown.

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

    def seg_diff(self, seg1, seg2):
        """Return the features by which seg1 and seg2 differ.

        seg1, seg2 -- segments (lists of <value, name> pairs)
        """
        def seg_to_dict(seg):
            return {k: v for (v, k) in seg}
        assert seg_to_dict([(1, 2), (3, 4)]) == {1: 2, 3: 4}

    def fts_to_str(self, seg):
        vals = {u'0': ' ', u'-': '0', u'+': '1'}
        seg_dict = {n: v for (v, n) in seg}
        vector = []
        for name in self.names:
            if name in seg_dict:
                vector.append(vals[seg_dict[name]])
        return ''.join(vector)

    def fts_contrast(self, fs, ft_name, inv):
        """Return True if there is a segment in inv that contrasts in feature
        ft_name.

        ft_name -- name of the feature where contrast must be present.
        inv -- collection of segments represented as Unicode segments.
        """
        plus, minus = (u'+', ft_name), (u'-', ft_name)
        w_plus, w_minus = set(list(fs) + [plus]), set(list(fs) + [minus])
        return any([self.fts_match(w_plus, s) for s in inv]) and \
               any([self.fts_match(w_minus, s) for s in inv])

    def fts_count(self, fts, inv):
        """Returns the count of segments in an inventory matching a give
        feature mask.

        fts -- feature mask given as a set of <val, name> tuples
        inv -- inventory of segments (as Unicode IPA strings)
        """
        return len(filter(lambda s: self.fts_match(fts, s), inv))
