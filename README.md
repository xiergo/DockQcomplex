# DockQcomplex
Calculate DockQ for multimer, defined as the average of DockQ values for all interfaces of paired chains in contact.


# Method
Each of the predicted chains were greedily assigned to their nearest neighbour of the same sequence in the ground truth, as suggusted in AF-Multimer (refer to AF-Multimer SI Part 7.3.2). Two chains were defined as contacted as long as any heavy atom of one chain being within 5A of any heavy atom of the other chain. All chain pair with contacts were used to calculate DockQ in 'Two chains (Dimer)' mode with the code in (https://github.com/bjornwallner/DockQ). The averaged DockQ value was defined as the final DockQ for the protein complex.

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
For example:
```
python dockq_complex.py example/7URD/relaxed_model_1_multimer_v3_pred_0.pdb example/7URD/ 7URD
```
All intermediate files can be found in `_tmp/[pdb_id]`.

# Reference
[DockQ: A Quality Measure for Protein-Protein Docking Models](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879)

[Protein complex prediction with AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)
