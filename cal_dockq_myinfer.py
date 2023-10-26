import os
import warnings

import pandas as pd
import numpy as np
import multiprocessing as mp
from dockq_complex import cal_dockq_pdb

model_dict = {'model2': 'ft-64-alink_step_3000_10',
              'model1': 'ft-64_step_17000_10'}


def cal_dockq(pdb_id, inputdir):
    pred_pdb = f'../myinfer_res/{inputdir}/{pdb_id}_pred.pdb'# ft-64-alink_step_3000_10/
    if not os.path.isfile(pred_pdb):
        return None, None
    try:
        dockq, rmsd = cal_dockq_pdb(pred_pdb, '../pdb_no_gap/', pdb_id)
    except Exception as e:
        dockq = None
        rmsd = None
        print(f'fail {pdb_id}')
        print(e)
    return dockq, rmsd


def main():
    df = pd.read_csv('testset_info_3300_dockq_update.tsv', sep='\t')
    for model, inputdir in model_dict.items():
        num_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        with mp.Pool(num_of_cores) as pool:
            res = pool.starmap(cal_dockq, [(pdb_id, inputdir) for pdb_id in df.pdb_id])
        df[f'dockq_{model}'] = [i[0] for i in res]
        df[f'rmsd_{model}'] = [i[1] for i in res]
    df.to_csv(f'testset_info_3300_dockq_update_update.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()