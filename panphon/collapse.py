from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
from importlib.resources import files

import yaml

from panphon import _panphon, permissive


class Collapser(object):
    def __init__(
            self, tablename='dogolpolsky_prime.yml', feature_set='spe+', feature_model='strict'
    ):
        fm = {'strict': _panphon.FeatureTable,
              'permissive': permissive.PermissiveFeatureTable}
        self.fm = fm[feature_model](feature_set=feature_set)
        self.rules = self._load_table(tablename)

    def _load_table(self, tablename: str):
        fn = os.path.join('data', tablename)
        with files('panphon').joinpath(fn).open('r') as f:
            rules = []
            table = yaml.load(f.read(), Loader=yaml.FullLoader)
            for rule in table:
                rules.append((_panphon.fts(rule['def']), rule['label']))
        return rules

    def collapse(self, s):
        segs = []
        for seg in self.fm.seg_regex.findall(s):
            fts = self.fm.fts(seg)
            for mask, label in self.rules:
                if self.fm.match(mask, fts):
                    segs.append(label)
                    break
        return ''.join(segs)
