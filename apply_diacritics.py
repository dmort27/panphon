#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script for applying diacritics and modifiers to IPA characters
# systemically, based on feature arrays.
# David R. Mortensen, 2014-07-2

import argparse
import unicodedata
import unicodecsv as csv
import codecs
import yaml
import sys
import regex


# Regular expressions

letter_re = regex.compile(ur"""([\p{Block=BasicLatin}
                                \p{Block=GreekAndCoptic}
                                \p{Block=IPAExtensions}œ\u00C0-\u00FF]
                                [\u0300-\u0360\u0362-\u036F]*)""", regex.X)


# Operations on feature structures.


def feature_match(condition, array):
    """Takes a condition and a complete array of features, both as
    dictionaries of feature names associated with feature values, and
    determines whether the condition is contained within the
    array. Returns a boolean value.
    """
    if array is None:
        return False
    else:
        return set(condition.iteritems()).issubset(set(array.iteritems()))


def feature_update(content, array):
    """Updates a feature array (really a dictionary of feature names
    associated with feature values) with values from content, a
    dictionary containing specifications for a subset of the features
    in array.
    """
    if array is None:
        return None
    else:
        new_array = array.copy()
        for (ft, val) in content.iteritems():
            new_array[ft] = val
        return new_array


def vectorize_features(ft_values, ft_names):
    """Takes a dictionary of feature names associated with feature
    values and a list of feature names; returns a list of feature
    values in the same order as fnames.
    """
    return [ft_values[name] for name in ft_names]


# File operations.


def parse_segments_file(segment_fn):
    """Parses a UTF-8 CSV file containing IPA segments in the first
    column, feature names in the first row, and corresponding feature
    values in the remainder of the cells.
    """
    segs = []
    with open(segment_fn, "rb") as f:
        reader = csv.reader(f, encoding="utf-8")
        ft_names = reader.next()[1:]
        for row in reader:
            seg, ft_vals = row[0], row[1:]
            segs.append((seg, dict(zip(ft_names, ft_vals))))
    return (segs, ft_names)


def parse_simple_segments_file(fn):
    """Parses a CSV file with one column. Returns a list of the
    contents of that column.
    """
    segs = []
    with open(fn, "rb") as f:
        reader = csv.reader(f, encoding="utf-8")
        for [item] in reader:
            segs.append(item)
    return segs


def parse_diacritic_file(diacritic_fn):
    """Parses a UTF-8 YAML file describing diacritics/modifiers.
    """
    return yaml.load(codecs.open(diacritic_fn, "r", "utf-8").read())


def write_features_file(segment_fn, ft_names, segments):
    """Writes a UTF-8 encoded CSV file containing IPA segments in the
    first column, feature names in the first row, and corresponding
    feature values in the remainder of the cells.
    """
    with open(segment_fn, "wb") as f:
        writer = csv.writer(f, encoding="utf-8")
        writer.writerow(["IPA Segment"] + ft_names)
        for seg in segments:
            if seg:
                (s, v) = seg
                row = [s] + vectorize_features(v, ft_names)
            else:
                row = [""] + ([""] * len(ft_names))
            writer.writerow(row)


def write_seg_series_file(fn, seg_series, series_names):
    """Writes a UTF-8 encoded CSV file in which the first row contains
    a list of names and the subsequent rows contain the contents of a
    list of lists.
    """
    with open(fn, "wb") as f:
        writer = csv.writer(f, encoding="utf-8")
        writer.writerow(series_names)
        length = len(seg_series[0])
        for i in range(length):
            row = []
            for series in seg_series:
                row.append(series[i])
            writer.writerow(row)


def write_seg_sets(fn, segs):
    with open(fn, "wb") as f:
        writer = csv.writer(f, encoding="utf-8")
        for seg in segs:
            writer.writerow([seg])


class DiacriticApplier():
    """A class to apply diacritics/modifiers to IPA segments based on
    feature definitions.
    """

    def __init__(self, (segments, ft_names), diacritics, verbose):
        """The constructor takes two arguments, a tuple consisting of
        the segment dictionary and a list of feature names, and a list
        of dictionaries describing diacritics/modifiers.
        """
        self.ft_names = ft_names
        self.segments = segments
        self.diacritics = diacritics["diacritics"]
        self.combinations = diacritics["combinations"]
        self.series_names = [dia["name"] for dia in self.diacritics] + \
                            [com["name"] for com in self.combinations]
        self.verbose = verbose

    def sort_segments(self, sort_order):
        """Takes a list of dictionaries each containing an entry
        "name", containing a feature name, and an entry "reversed",
        which specifies that the sort order should be reverse, and
        uses this information to sort self.segments.
        """
        for ft in reversed(sort_order):
            if self.verbose:
                print "Sorting segments by {} {}...".format(
                    ft["name"],
                    "reversed" if ft["reverse"] else "unreversed")
            self.segments.sort(key=lambda (seg, fts):
                               fts[ft["name"]],
                               reverse=ft["reverse"])
        if self.verbose:
            print "Sort complete."

    def apply_diacritic_to_segment(self, (seg, array), diacritic):
        """Takes a tuple describing a segment, consisting of a string
        and a feature array (dictionary), and a diacritic, consisting
        of a dictionary with other nested data structures, and returns
        the result of applying the diacritic to the segment. If it
        cannot be applied, the method returns (None, None).
        """

        if self.verbose:
            print u"Applying ◌{} ({}) to segment {}.".format(
                diacritic["marker"], diacritic["name"], seg)

        exclude = diacritic["exclude"] if "exclude" in diacritic else []

        match = any([feature_match(cond, array)
                     for cond
                     in diacritic["conditions"]])
        if match and seg not in exclude:
            if self.verbose:
                print u" Conditions met for ◌{} ({}) on segment {}".format(
                    diacritic["marker"], diacritic["name"], seg)
            if diacritic["position"] == "pre":
                seg = diacritic["marker"] + seg
            else:
                if unicodedata.category(diacritic["marker"]) == "Lm":
                    seg = seg + diacritic["marker"]
                else:
                    seg = letter_re.sub("\g<1>" + diacritic["marker"], seg)
            array = feature_update(diacritic["content"], array)
            return (seg, array)
        else:
            if self.verbose:
                print u" Conditions NOT met for ◌{} ({}) on segment {}".format(
                    diacritic["marker"], diacritic["name"], seg)
            return (None, None)

    def apply_combination_to_segment(self, (seg, array), combination):
        """Takes a tuple describing a segment, consisting of a string
        and a feature array (dictionary), and a dictionary describing
        a combination of diacritics (defined in
        self.diacritics). Returns the result of applying all
        diacritics in the combination to the segment. If any of the
        diacritics cannot be applied, the method returns None.
        """
        if self.verbose:
            print u"Applying combination '{}' to segment {}".format(
                combination["name"], seg)
        for dia_name in combination["combines"]:
            if self.verbose:
                print u"Applying combined diacritic '{}'".format(dia_name)
            [dia] = [d for d in self.diacritics if d["name"] == dia_name]
            (seg, array) = self.apply_diacritic_to_segment((seg, array), dia)
        return (seg, array)

    def apply_diacritic_to_series(self, dia):
        """Applies a diacritic to the whole series of segments stored
        in self.segments.
        """
        return [self.apply_diacritic_to_segment(seg, dia)
                for seg
                in self.segments]

    def apply_combination_to_series(self, combo):
        """Applies a combination to the whole series of segments
        stored in self.segments.
        """
        return [self.apply_combination_to_segment(seg, combo)
                for seg
                in self.segments]

    def apply_diacritic_to_series_by_marker(self, marker):
        """Takes a marker for a diacritic/modifier relationship as an
        argument and applies the diacritic transformation for that
        marker to the whole series of segments.
        """
        try:
            [diacritic] = [dia
                           for dia
                           in self.diacritics
                           if dia["marker"] == marker]
        except IndexError:
            raise KeyError("Unknown marker specified.")
        return self.apply_diacritic_to_series(diacritic)

    def apply_all_diacritics_to_series(self):
        """Applies each diacritic/modifier, including combinations, to
        the whole series of segments stored in self.segments.
        """
        seg_series_pl = [[s for (s, d) in self.segments]]
        new_segs = []
        for dia in self.diacritics:
            if self.verbose:
                print u"Applying marker ◌{} to series.".format(dia["marker"])
            segs = self.apply_diacritic_to_series(dia)
            new_series = [seg for (seg, array)
                          in map(lambda (s, a): (s, a)
                                 if s
                                 else ("", None), segs)]
            seg_series_pl.append(new_series)
            new_segs += filter(lambda (s, a): s, segs)

        for combo in self.combinations:
            if self.verbose:
                print u"Applying combinations {} to series.".format(
                    combo["name"])
            segs = self.apply_combination_to_series(combo)
            new_series = [seg
                          for (seg, array)
                          in map(lambda (s, a):
                                 (s, a) if s else ("", None), segs)]
            seg_series_pl.append(new_series)
            new_segs += filter(lambda (s, a): s, segs)
        self.segments += new_segs
        return seg_series_pl

    def compare_segments_to_external(self, external):
        """Compare the segments in self.segments to those in the list
        of strings, external. Returns a tuple of segments in
        self.segments and not in external and segments in external but
        not in self.segments.
        """
        internal_segs = set([unicodedata.normalize("NFD", seg)
                             for (seg, fts)
                             in self.segments])
        external_segs = set([unicodedata.normalize("NFD", seg)
                             for seg
                             in external])
        return (sorted(internal_segs.difference(external_segs)),
                sorted(external_segs.difference(internal_segs)))


def main():
    # Dangerous encoding magic
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)

    # Set up the parser for the command-line arguments
    parser = argparse.ArgumentParser(
        description="Apply diacritics to IPA segments.")
    parser.add_argument("segments",
                        help="CSV file containg IPA segment descriptions")
    parser.add_argument("diacritics",
                        help="YAML file describing diacritics/modifiers")
    parser.add_argument("-o", "--output",
                        help="Output file",
                        default="segment_series.csv")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("-m", "--marker",
                       help="Apply the diacritic or modifier specified")
    scope.add_argument("-a", "--all",
                       help="Apply all diacritics/modifiers and output file containing all resulting series",
                       action="store_true")
    parser.add_argument("-f", "--features",
                        help="Write segment definitions to the specified file.")
    parser.add_argument("-v", "--verbose",
                        help="Turn on verbose debugging behavior.",
                        action="store_true")
    parser.add_argument("-s", "--sort",
                        help="Sort segments based on YAML list of features.")
    parser.add_argument("-c", "--compare",
                        help="Compare an external segment file to the internally generated segment list. Writes two output files.",
                        nargs=3)
    args = parser.parse_args()

    dia_app = DiacriticApplier(parse_segments_file(args.segments),
                               parse_diacritic_file(args.diacritics),
                               args.verbose)

    # If the sort option is set, sort twice--once here, before
    # computing the contents of the segment series file and once
    # later, before writing the features file.
    if args.sort:
        sort_order = yaml.load(codecs.open(args.sort, "r", "utf-8").read())
        dia_app.sort_segments(sort_order)

    # If the marker option is set, only apply the specified
    # diacritic/modifier.
    if args.marker:
        segs = dia_app.apply_diacritic_to_series_by_marker(
            args.marker.decode("utf-8"))
        seg_series = [s
                      for (s, v)
                      in map(lambda x: x if x else ("", None), segs)]
        write_seg_series_file(args.output, [seg_series])

    # Most of the time, users are going to want the all option set, so
    # we should probably make it a default. For now, it's mututally
    # exclusive with the marker option.
    elif args.all:
        seg_series_pl = dia_app.apply_all_diacritics_to_series()
        write_seg_series_file(args.output,
                              seg_series_pl,
                              ["Pure"] + dia_app.series_names)

    # Special function for comparing our output to other sources of
    # phoneme data (e.g. PHOIBLE). Really only makes sense with the
    # all option set.
    if args.compare:
        external = parse_simple_segments_file(args.compare[0])
        (pred_not_att, att_not_pred) = \
            dia_app.compare_segments_to_external(external)
        write_seg_sets(args.compare[1], pred_not_att)
        write_seg_sets(args.compare[2], att_not_pred)

    # This is intentionally the second sorting of the segments. It is
    # deliberately called after the addition of new segments (with
    # diacritics and modifiers) to dia_app.segments.
    if args.sort:
        sort_order = yaml.load(codecs.open(args.sort, "r", "utf-8").read())
        dia_app.sort_segments(sort_order)

    if args.features:
        write_features_file(args.features,
                            dia_app.ft_names,
                            dia_app.segments)

if __name__ == "__main__":
    main()
