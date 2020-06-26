import logging
from typing import List

import ase.data
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import Mol

from graphdg.parse.error import ConversionError
from .xyz2mol import xyz2mol

symbol_to_atomic_number = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
}

atomic_number_to_symbol = {v: k for k, v in symbol_to_atomic_number.items()}


def get_atom_distance(molecule: Mol, i1: int, i2: int, conf_id: int = 0) -> float:
    conformer = molecule.GetConformer(conf_id)
    vector = conformer.GetAtomPosition(i1) - conformer.GetAtomPosition(i2)
    return vector.Length()


def molecule_to_xyz(molecule: Mol, conf_id=0, comment=None) -> str:
    num_atoms = molecule.GetNumAtoms()
    string = f'{num_atoms}\n'

    if comment:
        string += comment

    conformer = molecule.GetConformer(conf_id)

    for atom_idx in range(molecule.GetNumAtoms()):
        atom = molecule.GetAtomWithIdx(atom_idx)
        position = conformer.GetAtomPosition(atom_idx)
        string += f'\n{atom.GetSymbol()} {position.x} {position.y} {position.z}'

    return string


def molecule_to_xyz_file(molecule: Mol, path: str, conf_id=0) -> None:
    xyz_string = molecule_to_xyz(molecule=molecule, conf_id=conf_id)

    with open(path, mode='w') as f:
        f.write(xyz_string)


def atoms_to_molecule(atoms: Atoms) -> Mol:
    data = {
        'atoms': [ase.data.atomic_numbers[atom.symbol] for atom in atoms],
        'coordinates': atoms.positions,
        'charge': 0,
        'allow_charged_fragments': False,
        'use_graph': True,
        'use_huckel': False,
        'embed_chiral': True,
    }

    try:
        return xyz2mol(**data)  # type: ignore
    except Exception as e:
        raise ConversionError(e)


def select_parsable_atoms(atoms_list: List[Atoms]) -> List[Mol]:
    molecules = []
    for index, atoms in enumerate(atoms_list):
        try:
            mol = atoms_to_molecule(atoms)
            frags = Chem.GetMolFrags(mol)
            if len(frags) == 1:
                mol.SetProp('id', str(index + 1))
                molecules.append(mol)
        except ConversionError:
            pass

    logging.info(f'Successfully parsed {len(molecules)}/{len(atoms_list)} molecules')
    return molecules
