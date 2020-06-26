from unittest import TestCase

from rdkit import Chem

from graphdg.embed import get_init_bounds_matrix, embed_bounds_matrix


class TestEmbedding(TestCase):
    def test_init_bounds(self):
        mol = Chem.AddHs(Chem.MolFromSmiles('C'))
        bounds_matrix = get_init_bounds_matrix(mol)

        self.assertEqual(bounds_matrix.shape, (5, 5))

        for i in range(5):
            self.assertAlmostEqual(bounds_matrix[i, i], 0.0)

    def test_methane(self):
        mol = Chem.AddHs(Chem.MolFromSmiles('C'))
        bounds_matrix = get_init_bounds_matrix(mol)

        delta = 0.005
        ch_bond = 1.08
        hh_dist = 1.77

        # 4 edges
        for i in range(1, 5):
            bounds_matrix[0, i] = ch_bond + delta
            bounds_matrix[i, 0] = ch_bond - delta

        # 5 edges
        for (i, j) in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]:
            bounds_matrix[i, j] = hh_dist + delta
            bounds_matrix[j, i] = hh_dist - delta

        i = embed_bounds_matrix(mol, bounds_matrix)

        self.assertEqual(0, i)
