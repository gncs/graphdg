from unittest import TestCase

from graphdg.parse.distances import EdgeInfo


class TestDistances(TestCase):
    def test_distance(self):
        distance = 1.5
        atom_ids = (1, 3)

        d = EdgeInfo(
            distance=distance,
            atom_ids=atom_ids,
            kind=1,
        )

        ids, feats, distance = d.to_features()

        self.assertTupleEqual(ids, atom_ids)
        self.assertEqual(len(feats), 21)
