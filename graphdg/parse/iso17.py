import logging
import os
from typing import List

from ase import Atoms
from ase.db import connect
"""
The data is partitioned as used in the SchNet paper [6] (arXiv:1706.08566):                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                              
reference.db       - 80% of steps of 80% of MD trajectories                                                                                                                                                                                                                                                                   
reference_eq.db    - equilibrium conformations of those molecules                                                                                                                                                                                                                                                             
test_with.db       - remaining 20% unseen steps of reference trajectories                                                                                                                                                                                                                                                     
test_other.db      - remaining 20% unseen MD trajectories                                                                                                                                                                                                                                                                     
test_eq.db         - equilibrium conformations of test trajectories  
"""


def parse_dataset(path: str) -> List[Atoms]:
    file_names = ['test_other.db', 'reference.db', 'test_within.db']
    logging.info('Parsing ISO17 data set files: ' + str(file_names))

    items = []
    for file_name in file_names:
        items += parse_db(path=os.path.join(path, file_name))

    logging.info(f'Parsed {len(items)} structures')
    return items


def parse_db(path) -> List[Atoms]:
    atoms_list = []
    with connect(path) as conn:
        for row in conn.select():
            atoms_list.append(row.toatoms())

    return atoms_list
