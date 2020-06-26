from distutils.util import convert_path
from typing import Dict

from setuptools import setup, find_packages


def readme() -> str:
    with open('README.md') as f:
        return f.read()


version_dict = {}  # type: Dict[str, str]
with open(convert_path('graphdg/version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='graphdg',
    version=version_dict['__version__'],
    description='GraphDG - A Deep Generative Model for Molecular Distance Geometry based on Graph Neural Networks',
    long_description=readme(),
    classifiers=['Programming Language :: Python :: 3.7'],
    author='Gregor Simm',
    author_email='gncs2@cam.ac.uk',
    python_requires='>=3.7',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'graph_nets',
        'networkx',
        'ase',
    ],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
)
