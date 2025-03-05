# EvalComplex
Calculate DockQ for multimer, defined as the average of DockQ values for all interfaces of paired chains in contact.


# Method
The predicted chains were assigned to their nearest neighbor of the same sequence in the ground truth in a greedy manner, as suggested in AF-Multimer (please refer to AF-Multimer SI Part 7.3.2). Two chains were considered in contact if any heavy atom of one chain was within 5A of any heavy atom of the other chain. The DockQ was calculated for all chain pairs with contacts in 'Two chains (Dimer)' mode using the code from (https://github.com/bjornwallner/DockQ). The final DockQ for the protein complex was defined as the average DockQ value.

# Installation
```bash
git clone https://github.com/xiergo/EvalComplex.git
cd EvalComplex/dockq
sh clean.bash
make
cd ..
```

Python packages `numpy`, `pandas`, and `biopython` were required. Please install them first by directly running:
```bash
pip install numpy
pip install pandas
pip install biopython
```

# Usage

Type `python dockq_complex.py -h` to view usage page:

```
usage: dockq_complex.py [-h] [--pdb_id PDB_ID] [--key KEY] [--debug_mode] pred_pdb_path truth_pdb_path

Calculate DockQ for protein complex

positional arguments:
  pred_pdb_path    a pdb file containing predicted structures of all chains
  truth_pdb_path   a directory containing all ground truth pdb files, with each file corresponding to one chain, or it can also be a pdb file consisting of all chains

optional arguments:
  -h, --help       show this help message and exit
  --pdb_id PDB_ID  PDB id, if "truth_pdb_path" is a directory, all files in "truth_pdb_path" with the pattern "pdb_id***pdb" (excluding pred_pdb_path) will be recognized as ground truth pdbs
  --key KEY        output directory identifier
  --debug_mode     it will print and save intermediate results with this mode on

```

This is an example:
```
python dockq_complex.py example/7URD/relaxed_model_1_multimer_v3_pred_0.pdb example/7URD/ --pdb_id 7URD

python dockq_complex.py example/7Y8U_pred.pdb example/7Y8U_truth.pdb
```

All intermediate output files can be found in `_tmp/[pdb_id]_[key]_[date-time]_[5_digits_random_number]` in debug mode.


Note that:
1. Ground truth chains should be saved in separate pdb files in one directory or one pdb file. If it is in the former case, you need to provide 'pdb_id'. In addition, each pdb file include only one chain of the complex and should be named as '(pdb_id)_(chain_id).pdb', where chain id can be anything with any length, not necessarily consistent with chain id in prediction, such as '7URD_I_am_one_chain.pdb' and '7URD_I_am_another_chain.pdb'.
2. The chain id can not match between prediction and truth, but the residue index must stay consistent between prediction and truth. Gaps are allowed in ground truth, and will be inserted 'UNK(X)' residue placeholders automatically to make the same length between counterpart chain of prediction and truth.

# Reference
1. [DockQ: A Quality Measure for Protein-Protein Docking Models](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879)

2. [Protein complex prediction with AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)
