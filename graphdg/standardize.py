from typing import Tuple, Dict, List

import numpy as np
from graph_nets.graphs import GraphsTuple

from .tf_tools import graphs_tuple_to_data_dicts, data_dicts_to_graphs_tuple

MIN_STD = 1E-6


class Standardizer:
    @staticmethod
    def compute_mean_std(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.mean(a, axis=0),
            np.maximum(np.std(a, axis=0), MIN_STD),
        )

    @staticmethod
    def _standardize(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (a - mean) / std

    @staticmethod
    def _destandardize(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (a * std) + mean


class ArrayStandardizer(Standardizer):
    def __init__(
            self,
            mean: np.ndarray = np.array([0]),
            std: np.ndarray = np.array([1]),
    ):
        self.mean, self.std = mean, std

    def standardize(self, array) -> np.ndarray:
        return self._standardize(a=array, mean=self.mean, std=self.std)

    def destandardize(self, array) -> np.ndarray:
        return self._destandardize(a=array, mean=self.mean, std=self.std)

    @classmethod
    def from_array(cls, array: np.ndarray):
        mean, std = cls.compute_mean_std(a=array)
        return cls(mean=mean, std=std)


class GraphStandardizer(Standardizer):
    def __init__(
            self,
            global_mean: np.ndarray = np.array([0]),
            global_std: np.ndarray = np.array([1]),
            nodes_mean: np.ndarray = np.array([0]),
            nodes_std: np.ndarray = np.array([1]),
            edges_mean: np.ndarray = np.array([0]),
            edges_std: np.ndarray = np.array([1]),
    ):
        self.global_mean, self.global_std = global_mean, global_std
        self.nodes_mean, self.nodes_std = nodes_mean, nodes_std
        self.edges_mean, self.edges_std = edges_mean, edges_std

    def standardize_graphs_tuple(self, graphs: GraphsTuple) -> GraphsTuple:
        standard = graphs.replace(globals=self._standardize(graphs.globals, mean=self.global_mean, std=self.global_std))
        standard = standard.replace(nodes=self._standardize(graphs.nodes, mean=self.nodes_mean, std=self.nodes_std))
        standard = standard.replace(edges=self._standardize(graphs.edges, mean=self.edges_mean, std=self.edges_std))

        return standard

    def standardize_data_dict(self, d: Dict) -> Dict:
        return graphs_tuple_to_data_dicts(self.standardize_graphs_tuple(data_dicts_to_graphs_tuple([d])))[0]

    def destandardize_graphs_tuple(self, graphs: GraphsTuple) -> GraphsTuple:
        standard_graphs = graphs.replace(
            globals=self._destandardize(graphs.globals, mean=self.global_mean, std=self.global_std))
        standard_graphs = standard_graphs.replace(
            nodes=self._destandardize(graphs.nodes, mean=self.nodes_mean, std=self.nodes_std))
        standard_graphs = standard_graphs.replace(
            edges=self._destandardize(graphs.edges, mean=self.edges_mean, std=self.edges_std))

        return standard_graphs

    def destandardize_data_dicts(self, d: Dict) -> Dict:
        return graphs_tuple_to_data_dicts(self.destandardize_graphs_tuple(data_dicts_to_graphs_tuple([d])))[0]

    @classmethod
    def from_graphs_tuple(cls, graphs_tuple: GraphsTuple):
        global_mean, global_std = cls.compute_mean_std(graphs_tuple.globals)
        nodes_mean, nodes_std = cls.compute_mean_std(graphs_tuple.nodes)
        edges_mean, edges_std = cls.compute_mean_std(graphs_tuple.edges)

        return cls(
            global_mean=global_mean,
            global_std=global_std,
            nodes_mean=nodes_mean,
            nodes_std=nodes_std,
            edges_mean=edges_mean,
            edges_std=edges_std,
        )

    @classmethod
    def from_data_dicts(cls, dicts: List[Dict]):
        return cls.from_graphs_tuple(data_dicts_to_graphs_tuple(dicts))
