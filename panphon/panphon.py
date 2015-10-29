# -*- coding: utf-8 -*-

import pkg_resources
import unicodecsv as csv


class FeatureError(Exception):
    pass


class SegmentError(Exception):
    pass


class FeatureTable(object):
    """Encapsulate the segment <=> feature mapping in the file
    data/segment_features.csv.

    """
    def __init__(self):
        self._read_table()
        # Sanity checks for self.feature_match
        assert self.feature_match(
            set([(u'+', u'sg'), (u'-', u'syl'), (u'-', u'cor')]), u'pʰ')
        assert self.feature_match(
            set([(u'-', u'tense'), (u'+', u'hi'), (u'-', u'back')]), u'ɪ')
        assert self.feature_match(
            set([(u'+', u'nas'), (u'-', u'voi'), (u'+', u'cor')]), u'n̥')

    def _read_table(self):
        """Read the data from data/segment_features.csv into self.segments, a
        list of 2-tuples of unicode strings and sets of feature tuples
        and self.seg_dict, a dictionary mapping from unicode segments
        and sets of feature tuples.

        """
        filename = pkg_resources.resource_filename(
            __name__, 'data/segment_features.csv')
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
        # A few sanity checks:
        assert (u'+', u'cons') in self.seg_dict[u'tʰ']
        assert set([(u'-', u'cons'), (u'-', u'hi')]) <= self.seg_dict[u'a']

    def feature_match(self, features, segment):
        """Evaluates whether a set of features 'match' a segment (are a subset
        of that segment's features); returns 'None' if segment is unknown.
        """
        features = set(features)
        if segment in self.seg_dict:
            return features <= self.seg_dict[segment]
        else:
            return None

    def segment_known(self, segment):
        """Returns True if segment is in segment <=> features database."""
        return segment in self.seg_dict

    def segment_features(self, segment):
        """Returns the features as a list of 2-tuples, given a segment as a
        Unicode string; returns 'None' if segment is unknown.

        segment -- segment for which features are to be returned as
        Unicode string """
        if segment in self.seg_dict:
            return self.seg_dict[segment]
        else:
            return None

    def feature_intersection(self, segments):
        """Returns the features shared by all segments in the list/set of
        segments. Segments that are not known are ignored error.

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

    def features_match_any(self, fts, inv):
        """Returns a boolean based on whether there is a segment in 'inv'
        that matches all of the features in 'features'.

        features -- a collection of feature 2-tuples <val, name>
        inv -- a collection of segments represented as Unicode
               strings

        """
        return any([self.feature_match(fts, s) for s in inv])

    def features_match_all(self, fts, inv):
        """Returns a boolean based on whether all segments in 'inv'
         matche all of the features in 'features'.

        features -- a collection of feature 2-tuples <val, name>
        inv -- a collection of segments represented as Unicode
               strings

        """
        return all([self.feature_match(fts, s) for s in inv])
