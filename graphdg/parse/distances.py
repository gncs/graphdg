from dataclasses import dataclass
from typing import Tuple, List

import ase.data
import numpy as np
from networkx import Graph
from rdkit import Chem

import graphdg.parse.tools as tools
from graphdg.parse.extended_graph import mol_to_extended_graph
from graphdg.tools import GLOBALS, NODES, EDGES, SENDERS, RECEIVERS, to_one_hot

# Elements
SYMBOLS = ase.data.chemical_symbols[1:10]
SYMBOL_TO_OH = {symbol: to_one_hot(index=index, num_classes=len(SYMBOLS)) for index, symbol in enumerate(SYMBOLS)}

# Rings
MAX_RING_SIZE = 9
RING_SIZES = range(3, MAX_RING_SIZE + 1)
NOT_IN_RING = tuple(0 for _ in RING_SIZES)

# Edges
EDGE_KINDS = [1, 2, 3]
EDGE_KIND_TO_OH = {
    edge_kind: to_one_hot(index=index, num_classes=len(EDGE_KINDS))
    for index, edge_kind in enumerate(EDGE_KINDS)
}

# Bond type
BOND_TYPES = [
    Chem.BondType.ZERO, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC
]
BOND_TYPE_TO_OH = {
    bond_type: to_one_hot(index=index, num_classes=len(BOND_TYPES))
    for index, bond_type in enumerate(BOND_TYPES)
}

# Stereo
STEREO_TYPES = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY, Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]
STEREO_TYPE_TO_OH = {
    stereo_type: to_one_hot(index=index, num_classes=len(STEREO_TYPES))
    for index, stereo_type in enumerate(STEREO_TYPES)
}

# Chirality
CHI_TAGS = [Chem.CHI_UNSPECIFIED, Chem.CHI_TETRAHEDRAL_CW, Chem.CHI_TETRAHEDRAL_CCW]
CHI_TAGS_TO_OH = {chi_tag: to_one_hot(index=index, num_classes=len(CHI_TAGS)) for index, chi_tag in enumerate(CHI_TAGS)}


@dataclass
class NodeInfo:
    symbol: str
    chiral_tag: int

    def to_features(self) -> List:
        return SYMBOL_TO_OH[self.symbol] + CHI_TAGS_TO_OH[self.chiral_tag]


@dataclass
class EdgeInfo:
    distance: float
    atom_ids: Tuple[int, int]
    kind: int
    stereo: int = Chem.BondStereo.STEREONONE
    bond_type: int = Chem.BondType.ZERO
    is_aromatic: bool = False
    is_conjugated: bool = False
    is_in_ring_size: Tuple[int, ...] = NOT_IN_RING

    def to_features(self) -> Tuple[Tuple, List[int], float]:
        feats = EDGE_KIND_TO_OH[self.kind] + STEREO_TYPE_TO_OH[self.stereo] + BOND_TYPE_TO_OH[self.bond_type] + [
            int(self.is_aromatic), int(self.is_conjugated)
        ] + list(self.is_in_ring_size)
        return self.atom_ids, feats, self.distance


def get_node_infos(molecule: Chem.Mol) -> List[NodeInfo]:
    return [
        NodeInfo(
            symbol=ase.data.chemical_symbols[atom.GetAtomicNum()],
            chiral_tag=atom.GetChiralTag(),
        ) for atom in molecule.GetAtoms()
    ]


def get_edge_infos(molecule: Chem.Mol, graph: Graph):
    edge_infos = []
    for (source, sink) in graph.edges:
        kind = graph.edges[(source, sink)]['kind']

        if kind == 1:
            bond = molecule.GetBondBetweenAtoms(source, sink)
            edge_info = EdgeInfo(
                distance=tools.get_atom_distance(molecule, source, sink),
                atom_ids=(source, sink),
                kind=kind,
                stereo=bond.GetStereo(),
                bond_type=bond.GetBondType(),
                is_aromatic=bond.GetIsAromatic(),
                is_conjugated=bond.GetIsConjugated(),
                is_in_ring_size=tuple(int(bond.IsInRingSize(size)) for size in RING_SIZES),
            )
        else:
            edge_info = EdgeInfo(
                distance=tools.get_atom_distance(molecule, source, sink),
                atom_ids=(source, sink),
                kind=kind,
            )

        edge_infos.append(edge_info)

    return edge_infos


def get_feats_and_targets(node_infos: List[NodeInfo], edge_infos: List[EdgeInfo]) -> Tuple[dict, np.ndarray]:
    nodes = [node_info.to_features() for node_info in node_infos]
    edges = []
    senders = []
    receivers = []
    targets = []

    for edge_info in edge_infos:
        (sender, receiver), edge_feats, distance = edge_info.to_features()

        # Forward
        edges.append(edge_feats)
        senders.append(sender)
        receivers.append(receiver)
        targets.append([distance])

        # Reverse
        edges.append(edge_feats)
        senders.append(receiver)
        receivers.append(sender)
        targets.append([distance])

    assert (len(edges) == len(senders) == len(receivers) == len(targets))

    feats = {
        GLOBALS: np.array([], dtype=np.float),
        NODES: np.array(nodes, dtype=np.float),
        EDGES: np.array(edges, dtype=np.float),
        SENDERS: np.array(senders, dtype=np.int),
        RECEIVERS: np.array(receivers, dtype=np.int),
    }

    targets = np.array(targets, dtype=np.float)

    return feats, targets


def get_info_tuple(molecule: Chem.Mol, seed: int) -> Tuple[List[NodeInfo], List[EdgeInfo]]:
    atom_infos = get_node_infos(molecule)

    graph = mol_to_extended_graph(molecule, seed=seed)
    edge_infos = get_edge_infos(molecule=molecule, graph=graph)

    return atom_infos, edge_infos


def get_dataset(molecules: List[Chem.Mol], seed: int, count: int) -> List[Tuple[dict, np.ndarray]]:
    dataset = []
    for mol_id, molecule in enumerate(molecules):
        for index in range(count):
            node_infos, edge_infos = get_info_tuple(molecule, seed=seed + mol_id + index)
            feats_targets = get_feats_and_targets(node_infos=node_infos, edge_infos=edge_infos)
            dataset.append(feats_targets)

    return dataset
