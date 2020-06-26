import logging
import re
import tarfile
from typing import List

from ase import Atoms

from graphdg.parse.error import ParserError

coord_line = (br'(?P<element>\D+)\s+(?P<x>-?\d+\.\d*(E-?\d+)?)\s+(?P<y>-?\d+\.\d*(E-?\d+)?)\s+'
              br'(?P<z>-?\d+\.\d*(E-?\d+)?)\s+(?P<pcharge>-?\d+\.\d*(E-?\d+)?)\s*')
coord_re = re.compile(coord_line)
data_re = re.compile(
    br'^(?P<num_atoms>\d+)\n'
    br'gdb (?P<id>\d+)\s+(?P<A>-?\d+(\.\d*)?)\s+(?P<B>-?\d+\.\d*)\s+(?P<C>-?\d+\.\d*)\s+(?P<mu>-?\d+\.\d*)'
    br'\s+(?P<alpha>-?\d+\.\d*)\s+(?P<homo>-?\d+\.\d*)\s+(?P<lumo>-?\d+\.\d*)\s+(?P<gap>-?\d+\.\d*)'
    br'\s+(?P<r2>-?\d+\.\d*)\s+(?P<zpve>-?\d+\.\d*)\s+(?P<u_0>-?\d+\.\d*)\s+(?P<u_t>-?\d+\.\d*)'
    br'\s+(?P<h>-?\d+\.\d*)\s+(?P<g>-?\d+\.\d*)\s+(?P<cv>-?\d+\.\d*)\s+'
    br'(?P<coordinates>(' + coord_line + br')+)'
    br'(?P<vib_freqs>(\s*-?\d+\.\d*)+)'
    br'(?P<smiles_gdb17>(\s*\S+))'
    br'(?P<smiles_opt>(\s*\S+))'
    br'(?P<inchi_corina>(\s*\S+))'
    br'(?P<inchi_opt>(\s*\S+){2})\s*$')


def parse_entry(string: bytes) -> Atoms:
    match = data_re.match(string)
    try:
        if not match:
            raise ParserError('String does not match pattern')

        symbols = []
        positions = []

        for coord in coord_re.finditer(match.group('coordinates')):
            symbols.append(coord.group('element').decode('ascii').strip())
            positions.append([
                float(coord.group('x')),
                float(coord.group('y')),
                float(coord.group('z')),
            ])

        return Atoms(symbols=symbols, positions=positions)

    except (ValueError, AttributeError) as e:
        raise ParserError(e)


def parse_dataset(path: str) -> List[Atoms]:
    atoms_list = []
    logging.info('Parsing QM9 data set file: ' + str(path))

    with tarfile.open(path, mode='r') as archive:
        for i, entry in enumerate(archive):
            f = archive.extractfile(entry)

            if not f:
                raise ParserError('File cannot be read')

            string = f.read().replace(b'*^', b'E')

            try:
                atoms_list.append(parse_entry(string))
            except ParserError as e:
                logging.warning('Could not parse: ' + entry.name + ': ' + str(e))

    return atoms_list
