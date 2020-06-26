from unittest import TestCase

from rdkit import Chem

from graphdg.parse.distances import NodeInfo


class TestAtoms(TestCase):
    def test_atom(self):
        a = NodeInfo(
            symbol='H',
            chiral_tag=Chem.CHI_UNSPECIFIED,
        )

        feats = a.to_features()

        self.assertEqual(len(feats), 12)
