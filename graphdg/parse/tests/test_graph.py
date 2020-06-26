import collections
import os
from typing import Dict
from unittest import TestCase

import ase.io
import numpy as np
import pkg_resources

from graphdg.parse.extended_graph import mol_to_extended_graph, get_random_bf_sequence, build_bond_graph
from graphdg.parse.tools import atoms_to_molecule


class TestDistances(TestCase):
    RESOURCES = pkg_resources.resource_filename(__package__, 'resources')

    def test_sequence(self):
        xyz_path = os.path.join(self.RESOURCES, 'methane.xyz')
        atoms = ase.io.read(xyz_path, index=0, format='xyz')
        mol = atoms_to_molecule(atoms)

        rng = np.random.default_rng(seed=0)
        graph = build_bond_graph(mol)
        sequence = get_random_bf_sequence(graph, start=0, rng=rng)

        self.assertEqual(sequence[0], 0)
        self.assertEqual(len(sequence), mol.GetNumAtoms())

    def test_methane(self):
        xyz_path = os.path.join(self.RESOURCES, 'methane.xyz')
        atoms = ase.io.read(xyz_path, index=0, format='xyz')
        mol = atoms_to_molecule(atoms)

        graph = mol_to_extended_graph(mol, seed=0)
        self.assertEqual(graph.number_of_nodes(), mol.GetNumAtoms())
        self.assertTrue(len(graph.edges) >= 9)

        counter: Dict[int, int] = collections.Counter()
        for edge in graph.edges:
            counter[graph.edges[edge]['kind']] += 1

        self.assertEqual(counter[1], 4)
        self.assertEqual(counter[2], 6)
        self.assertEqual(counter[3], 0)

    def test_formic_acid(self):
        xyz_path = os.path.join(self.RESOURCES, 'formic_acid.xyz')
        atoms = ase.io.read(xyz_path, index=0, format='xyz')
        mol = atoms_to_molecule(atoms)

        graph = mol_to_extended_graph(mol, seed=0)
        self.assertEqual(graph.number_of_nodes(), mol.GetNumAtoms())
        self.assertEqual(len(graph.edges), 9)

        counter: Dict[int, int] = collections.Counter()
        for edge in graph.edges:
            counter[graph.edges[edge]['kind']] += 1

        self.assertEqual(counter[1], 4)
        self.assertEqual(counter[2], 4)
        self.assertEqual(counter[3], 1)

    def test_ethanol(self):
        xyz_path = os.path.join(self.RESOURCES, 'ethanol.xyz')
        atoms = ase.io.read(xyz_path, index=0, format='xyz')
        mol = atoms_to_molecule(atoms)

        num_atoms = mol.GetNumAtoms()

        graph = mol_to_extended_graph(mol, seed=0)
        self.assertEqual(graph.number_of_nodes(), num_atoms)
        self.assertTrue(len(graph.edges) >= 3 * num_atoms - 6)
