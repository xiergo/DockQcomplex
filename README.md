# DockQcomplex
Calculate DockQ for multimer, defined as the average of DockQ values for all interfaces of paired chains in contact.


# Method
The predicted chains were assigned to their nearest neighbor of the same sequence in the ground truth in a greedy manner, as suggested in AF-Multimer (please refer to AF-Multimer SI Part 7.3.2). Two chains were considered in contact if any heavy atom of one chain was within 5A of any heavy atom of the other chain. The DockQ was calculated for all chain pairs with contacts in 'Two chains (Dimer)' mode using the code from (https://github.com/bjornwallner/DockQ). The final DockQ for the protein complex was defined as the average DockQ value.

# Installation
```bash
git clone https://github.com/xiergo/DockQcomplex.git
cd DockQcomplex/dockq
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
```
usage: Calculate DockQ for protein complex [-h] pred_pdb truth_pdb_dir pdb_id

positional arguments:
  pred_pdb       a pdb file containing predicted structures of all chains
  truth_pdb_dir  a directory containing all ground truth pdb files, with each file corresponding to one chain
  pdb_id         PDB id, all files in "truth_pdb_dir" with the pattern "pdb_id***pdb" will be recognized as ground truth pdbs

optional arguments:
  -h, --help     show this help message and exit
```

Note that:
1. Ground truth chains should be saved in separate pdb files. Each pdb file should be named as '(pdb_id)_(chain_id).pdb', where chain id can be anything with any length, not necessarily consistent with chain id in prediction, such as '7URD_I_am_an_example.pdb'.
2. The sequence of ground truth should be the same as that of prediction, with gap residues represented by 'UNK'. You can add 'UNK' gaps for single chain pdb and get `**_no_gap.pdb` in the same directory as your input pdb file by running:
```bash
python add_gap_pdb.py path/to/pdb/to/add/gap sequence_length_of_resulted_pdb
```

For example:
```
python dockq_complex.py example/7URD/relaxed_model_1_multimer_v3_pred_0.pdb example/7URD/ 7URD
```

All intermediate files can be found in `_tmp/[pdb_id]`.



# Reference
[DockQ: A Quality Measure for Protein-Protein Docking Models](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879)

[Protein complex prediction with AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)
