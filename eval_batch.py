
import os
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import glob

sys.path.append('/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/EvalComplex')
from eval import eval
from utils import RunBatch, get_chain_id, generate_split, each_chain_start_from_one

 

def find_file(pdb_id, method, it):
    
    gt = glob.glob(f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/colabdock/CL_CSP_Release/*/{pdb_id}/groundtruth/{pdb_id}_native.pdb')[0]
    pred = {
        'AFM': f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/cl_csp/output_af2/{pdb_id}/relaxed_model_1_multimer_v3_pred_0.pdb',
        'GRASP': f'{grasp_outdir}/{pdb_id}_iter{it}.pdb'
    }
    split_file = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/cl_csp/cl_csp/split/{pdb_id}.txt'
    return gt, pred[method], split_file


class MyRunBatch(RunBatch):
    def __init__(self, ncpu, res_file, log_dir, colnames):
        super().__init__(ncpu, res_file, log_dir, colnames)

    def run_single(self, pdb_id, method, it, ckpt_id):
        truth, pred, split_file = find_file(pdb_id, method, it)
        print(truth, pred, split_file, sep='\n')
        cids = get_chain_id(truth)
        split = generate_split(split_file=split_file, cids=cids)
        rmsd, tmscore, dockq_avg, dockq_dimer = eval(pred, each_chain_start_from_one(truth), split)
        return rmsd, tmscore, dockq_avg, dockq_dimer


if __name__ == '__main__':

    # version = 'colabdock_ab_ft-grasp-v1-64_step_40000'
    # /dl/atlas_dls/1/24/xieyh/output/infer_grasp/cl_csp_dimer_iter_plddt60_dist10_ft-grasp-v1-64_step_88000
    version = 'cl_csp_iter_plddt60_dist10_ft-grasp-v1-64_step_88000'
    # /dl/atlas_dls/1/24/xieyh/output/infer_grasp/cl_csp_dimer_ft-grasp-v1-64_step_88000
    # /dl/atlas_dls/1/24/xieyh/output/infer_grasp/cl_csp_dimer_iter2_ft-grasp-v1-64_step_88000
    grasp_outdir = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/cl_csp/{version}/data'
    os.makedirs(grasp_outdir, exist_ok=True)
    scpfrom = f'root@ascend:/dl/atlas_dls/1/24/xieyh/output/infer_grasp/{version}/*'
    scp_file(scpfrom, grasp_outdir)
    log_dir = f'{grasp_outdir}/../log'
    os.makedirs(log_dir, exist_ok=True)

    
    pdb_ids = [i for i in os.listdir('/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/cl_csp/output_af2')]
    methods = ['AFM', 'GRASP']
    iters = [i+1 for i in range(10)]

    args = [(i, 'GRASP', k) for i in pdb_ids for k in iters if os.path.isfile(f'{grasp_outdir}/{i}_iter{k}.pdb')] + [(i, 'AFM', None) for i in pdb_ids]
    # print(args)

    ncpus = 8
    print(f'{ncpus} cpus are used')
    with mp.Pool(ncpus) as pool:
        res = pool.starmap(run_single, args)

    resfile = f'{grasp_outdir}/../results.tsv'
    df = pd.DataFrame(res, columns=['pdb_id', 'method', 'iter', 'rmsd', 'tmscore', 'dockq_avg', 'dockq_dimer'])
    df = df.sort_values(by=['pdb_id', 'method', 'iter'])
    df.to_csv(resfile, sep='\t', index=False)
    print(f'Results saved in {resfile}')

