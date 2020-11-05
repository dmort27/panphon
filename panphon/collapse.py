from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path

import pkg_resources
import yaml

from panphon import _panphon
from panphon import permissive


class Collapser(object):
    def __init__(self, tablename='dogolpolsky_prime.yml', feature_set='spe+', feature_model='strict'):
        """
        Initialize the feature table.

        Args:
            self: (todo): write your description
            tablename: (str): write your description
            feature_set: (todo): write your description
            feature_model: (str): write your description
        """
        fm = {'strict': _panphon.FeatureTable,
              'permissive': permissive.PermissiveFeatureTable}
        self.fm = fm[feature_model](feature_set=feature_set)
        self.rules = self._load_table(tablename)

    def _load_table(self, tablename):
        """
        Loads table.

        Args:
            self: (todo): write your description
            tablename: (str): write your description
        """
        fn = os.path.join('data', tablename)
        fn = pkg_resources.resource_filename(__name__, fn)
        with open(fn, 'r') as f:
            rules = []
            table = yaml.load(f.read(), Loader=yaml.FullLoader)
            for rule in table:
                rules.append((_panphon.fts(rule['def']), rule['label']))
        return rules

    def collapse(self, s):
        """
        Collapse a list of given segments.

        Args:
            self: (todo): write your description
            s: (todo): write your description
        """
        segs = []
        for seg in self.fm.seg_regex.findall(s):
            fts = self.fm.fts(seg)
            for mask, label in self.rules:
                if self.fm.match(mask, fts):
                    segs.append(label)
                    break
        return ''.join(segs)
