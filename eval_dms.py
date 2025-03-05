
import os
import sys
import glob

sys.path.append('/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/EvalComplex')
from eval import eval
from utils import RunBatch, get_chain_id, generate_split_ab, each_chain_start_from_one, scp_file


class MyRunBatch(RunBatch):
    def __init__(self, ncpu, res_file, log_dir):
        colnames = ['pdb_id', 'method', 'ckpt_id', 'it', 'rmsd', 'tmscore', 'dockq_avg', 'dockq_dimer']
        super().__init__(ncpu, res_file, log_dir, colnames)

    def run_single(self, pdb_id, method, ckpt_id, it):
        truth, pred, fasta = self.find_file(pdb_id, method, ckpt_id, it)
        print(truth, pred, fasta, sep='\n')
        split = generate_split_ab(fasta)
        rmsd, tmscore, dockq_avg, dockq_dimer = eval(pred, each_chain_start_from_one(truth), split)
        return rmsd, tmscore, dockq_avg, dockq_dimer


    def find_file(self, pdb_id, method, ckpt_id, it):
        gt = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/dms_data_full_antigen/data/pdb/{pdb_id}.pdb'
        pred = {
            'GRASP': f'{grasp_outdir}/ckpt_{ckpt_id}_{pdb_id}_iter{it}.pdb' # ckpt_2000_1YAG_iter3.pdb
        }
        fasta = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/dms_data_full_antigen/data/fasta/{pdb_id}.fasta'
        return gt, pred[method], fasta


if __name__ == '__main__':

    # /dl/atlas_dls/1/24/xieyh/output/infer_grasp_v2/ft-grasp-v2-32-notfixafm_dms_0.2_0
    version = sys.argv[1]
    grasp_outdir = f'dms/{version}/data'
    os.makedirs(grasp_outdir, exist_ok=True)
    scpfrom = f'root@ascend:/dl/atlas_dls/1/24/xieyh/output/infer_grasp_v7/{version}/*'
    scp_file(scpfrom, grasp_outdir)
    log_dir = f'{grasp_outdir}/../log'

    pdb_ids = [i.replace('_dms.tsv', '') for i in os.listdir('/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/dms_data_full_antigen/data/dms')]
    methods = ['GRASP']
    iters = [i+1 for i in range(10)]
    ckpt_ids = [i*1000 for i in range(100)]

    rb = MyRunBatch(ncpu=16, res_file=f'{grasp_outdir}/../results.tsv', log_dir=log_dir)

    args = [(pdb_id, method, ckpt_id, it) \
            for pdb_id in pdb_ids \
            for method in methods \
            for ckpt_id in ckpt_ids \
            for it in iters \
            if os.path.isfile(rb.find_file(pdb_id, method, ckpt_id, it)[1])]

    rb.run(args)
