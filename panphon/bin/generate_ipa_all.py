#!/usr/bin/env python
from __future__ import print_function, unicode_literals

import argparse
import codecs
import copy

import yaml

import unicodecsv as csv


class Segment(object):
    """Class modeling phonological segment."""

    def __init__(self, form, features):
        """Construct Segment objectself.

        Args:
            form (string): the segment as ipa
            features (list): the segment as feature_names
        """
        self.form = form
        self.features = features

    def __repr__(self):
        """Output string representation of Segment."""
        return 'Segment("{}", {})'.format(self.form,
                                          repr(self.features)).encode('utf-8')

    def feature_vector(self, feature_names):
        """Return feature vector for segment.

        Args:
            feature_names (list): ordered names of features

        Returns:
            list: feature values
        """
        return [self.features[ft] for ft in feature_names]


class Diacritic(object):
    """An object encapsulating a diacritics properties."""

    def __init__(self, marker, position, conditions, exclude, content):
        """Construct a diacritic object.

        Args:
            marker (unicode): the string form of the diacritic
            position (str): 'pre' or 'post', determining whether the diacritic
                            attaches before or after the base
            conditions (list): feature specification on which application of
                               diacritic is conditional
            exclude (list): conditions under which the diacritic will not be
                            applied]
            content (list): feature specification that will override
                            existing feature specifications when diacritics
                            is applied
        """
        self.marker = marker
        assert position in ['pre', 'post']
        self.position = position
        self.exclude = exclude
        self.conditions = conditions
        self.content = content

    def match(self, segment):
        if segment.form not in self.exclude:
            for condition in self.conditions:
                if set(condition.items()) <= set(segment.features.items()):
                    return True
            return False
        else:
            return False

    def apply(self, segment):
        if self.match(segment):
            new_seg = copy.deepcopy(segment)
            for k, v in self.content.items():
                new_seg.features[k] = v
            if self.position == 'post':
                new_seg.form = '{}{}'.format(new_seg.form, self.marker)
            else:
                new_seg.form = '{}{}'.format(self.marker, new_seg.form)
            return new_seg
        else:
            return None


class Combination(object):
    def __init__(self, diacritics, name, sequence):
        self.name = name
        self.sequence = [diacritics[d] for d in sequence]

    def apply(self, segment):
        new_seg = copy.deepcopy(segment)
        for dia in self.sequence:
            if dia.match(new_seg):
                new_seg = dia.apply(new_seg)
            else:
                return None
        return new_seg


def read_ipa_bases(ipa_bases):
    segments = []
    with open(ipa_bases, 'rb') as f:
        dictreader = csv.DictReader(f, encoding='utf=8')
        for record in dictreader:
            form = record['ipa']
            features = {k: v for k, v in record.items() if k != 'ipa'}
            segments.append(Segment(form, features))
    return segments


def parse_dia_defs(dia_defs):
    defs = yaml.load(codecs.open(dia_defs, "r", "utf-8").read(), Loader=yaml.FullLoader)
    diacritics = {}
    for dia in defs['diacritics']:
        if 'exclude' in dia:
            exclude = dia['exclude']
        else:
            exclude = []
        diacritics[dia['name']] = Diacritic(dia['marker'], dia['position'],
                                            dia['conditions'], exclude,
                                            dia['content'])
    combinations = []
    for comb in defs['combinations']:
        combinations.append(Combination(diacritics, comb['name'],
                                        comb['combines']))
    return diacritics, combinations


def sort_all_segments(sort_order, all_segments):
    all_segments_list = list(all_segments)
    field_order = reversed(yaml.load(open(sort_order, 'r').read(), Loader=yaml.FullLoader))
    for field in field_order:
        all_segments_list.sort(key=lambda seg: seg.features[field['name']],
                               reverse=field['reverse'])
    return all_segments_list


def write_ipa_all(ipa_bases, ipa_all, all_segments, sort_order):
    with open(ipa_bases, 'rb') as f:
        reader = csv.reader(f, encoding='utf-8')
        fieldnames = next(reader)
    with open(ipa_all, 'wb') as f:
        writer = csv.DictWriter(f, encoding='utf-8', fieldnames=fieldnames)
        writer.writerow({k: k for k in fieldnames})
        all_segments_list = sort_all_segments(sort_order, all_segments)
        for segment in all_segments_list:
            fields = copy.copy(segment.features)
            fields['ipa'] = segment.form
            writer.writerow(fields)


def main(ipa_bases, ipa_all, dia_defs, sort_order):
    segments = read_ipa_bases(ipa_bases)
    diacritics, combinations = parse_dia_defs(dia_defs)
    all_segments = set(segments)
    for diacritic in diacritics.values():
        for segment in segments:
            new_seg = diacritic.apply(segment)
            if new_seg is not None:
                all_segments.add(new_seg)
    for combination in combinations:
        for segment in segments:
            new_seg = combination.apply(segment)
            if new_seg is not None:
                all_segments.add(new_seg)
    write_ipa_all(ipa_bases, ipa_all, all_segments, sort_order)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bases', help='File containing IPA bases (ipa_bases.csv)')
    parser.add_argument('all', help='File to which all IPA segments is to be written (ipa_all.csv)')
    parser.add_argument('-d', '--dia', required=True, help='Diacritic definition file (default=diacritic_definitions.yml)')
    parser.add_argument('-s', '--sort-order', required=True, help='File definiting sort order.')
    args = parser.parse_args()
    main(args.bases, args.all, args.dia, args.sort_order)
