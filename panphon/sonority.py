from __future__ import print_function, absolute_import, unicode_literals

from . import _panphon
from . import permissive

from ._panphon import FeatureTable, fts


class BoolTree(object):
    def __init__(self, test=None, t_node=None, f_node=None):
        self.test = test
        self.t_node = t_node
        self.f_node = f_node

    def get_value(self):
        # logging.debug('t_node={} f_node={}'.format(self.t_node, self.f_node))
        if self.test:
            if isinstance(self.t_node, BoolTree):
                return self.t_node.get_value()
            else:
                # logging.debug('Returning {}'.format(self.t_node))
                return self.t_node
        else:
            if isinstance(self.f_node, BoolTree):
                return self.f_node.get_value()
            else:
                # logging.debug('Returning {}'.format(self.f_node))
                return self.f_node


class Sonority(object):
        def __init__(self, feature_set='spe+', feature_model='strict'):
            fm = {'strict': _panphon.FeatureTable,
                  'permissive': permissive.PermissiveFeatureTable}
            self.fm = fm[feature_model](feature_set=feature_set)

        def sonority_from_fts(self, seg):
            """Given a segment as features, returns the sonority on a scale of 1 to 9."""

            def match(m):
                return self.fm.match(fts(m), seg)

            minusHi = BoolTree(match('-hi'), 9, 8)
            minusNas = BoolTree(match('-nas'), 6, 5)
            plusVoi1 = BoolTree(match('+voi'), 4, 3)
            plusVoi2 = BoolTree(match('+voi'), 2, 1)
            plusCont = BoolTree(match('+cont'), plusVoi1, plusVoi2)
            plusSon = BoolTree(match('+son'), minusNas, plusCont)
            minusCons = BoolTree(match('-cons'), 7, plusSon)
            plusSyl = BoolTree(match('+syl'), minusHi, minusCons)
            return plusSyl.get_value()

        def sonority(self, seg):
            """Returns the sonority of a segment.

            seg -- segment given as a Unicode IPA string
            """
            return self.sonority_from_fts(self.fm.fts(seg))
