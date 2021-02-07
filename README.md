# GraphDG

<img src="resources/confs.png" width="40%">

**A Generative Model for Molecular Distance Geometry**<br>
Gregor N. C. Simm, José Miguel Hernández-Lobato <br>
*Proceedings of the 37th International Conference on Machine Learning*, Vienna, Austria, PMLR 119, 2020.<br>
https://arxiv.org/abs/1909.11459

## Installation

1. Create new Python 3.7 environment and activate:
   ```text
   virtualenv --python=python3.7 graphdg-venv
   source graphdg-venv/bin/activate
   ```

2. Install required packages and library itself:
   ```text
   pip install -r graphdg/requirements.txt
   pip install -e graphdg/
   ```

3. Install [RDKit (2020.03.1)](https://www.rdkit.org/docs/Install.html).

## Usage

1. Download and unpack ISO17 dataset
   ```text
   wget http://quantum-machine.org/datasets/iso17.tar.gz
   tar -xf iso17.tar.gz 
   ```

2. Prepare dataset
   ```text
   python3 graphdg/scripts/parse.py --path=iso17
   ```

3. Train model and generate conformations
   ```text
   python3 graphdg/scripts/run.py --train_path=iso17_split-0_train.pkl --test_path=iso17_split-0_test.pkl
   ```

## Reference

```text
@inproceedings{Simm2020GraphDG,
  title = 	 {A Generative Model for Molecular Distance Geometry},
  author = 	 {Simm, Gregor and Hernandez-Lobato, Jose Miguel},
  booktitle = 	 {Proceedings of the 37th International Conference on Machine Learning},
  pages = 	 {8949--8958},
  year = 	 {2020},
  editor = 	 {Hal Daumé III and Aarti Singh},
  volume = 	 {119},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--18 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v119/simm20a/simm20a.pdf},
  url = 	 {http://proceedings.mlr.press/v119/simm20a.html}
}
```

