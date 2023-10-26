import os
import warnings

import pandas as pd
import numpy as np
import multiprocessing as mp
from dockq_complex import cal_dockq_pdb


def cal_dockq(pdb_id):
    pred_pdbs = [i.strip() for i in os.popen(f'find ../output_af2/{pdb_id} -name relaxed_*.pdb').readlines()]
    pred_pdbs.sort()
    if not pred_pdbs:
        return None, None, None
    pred_pdb = pred_pdbs[-1]
    try:
        dockq, rmsd = cal_dockq_pdb(pred_pdb, '../pdb_no_gap/', pdb_id)
    except Exception as e:
        dockq = None
        rmsd = None
        print(f'fail {pdb_id}')
        print(e)
    return dockq, rmsd, pred_pdb


def main():
    df = pd.read_csv('../testset_info_3300.tsv', sep='\t')
    num_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    with mp.Pool(num_of_cores) as pool:
        res = pool.map(cal_dockq, df.pdb_id)
    df['dockq'] = [i[0] for i in res]
    df['rmsd'] = [i[1] for i in res]
    df['pred_path'] = [i[2] for i in res]
    df.to_csv('testset_info_3300_dockq_update.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()