import os
import sys
import glob
import pickle
import re
import pandas as pd
sys.path.append('/lustre/grp/gyqlab/share/xieyh/EvalComplex')
from eval import eval
import utils as utils



class MyRunBatch(utils.RunBatch):
    def __init__(self, ncpu, homedir, ana_name):
        self.ana_name = ana_name
        self.common_colnames = ['pdb_id', 'ckpt_id', 'seed', 'it']
        common_metric_colnames = ['rmsd', 'tmscore', 'dockq']
        if ana_name == '5jds':
            metric_colnames = common_metric_colnames
        elif ana_name == 'simxl':
            metric_colnames = common_metric_colnames + ['recall', 'recall_true']
        elif ana_name == 'dms' or ana_name == 'abxl':
            metric_colnames = common_metric_colnames + ['dockq_dimer']
        elif ana_name == 'cab':
            metric_colnames = ['dockq', 'rmsd']

        self.homedir = homedir
        self.grasp_outdir = f'{homedir}/data'
        res_file=f'{self.grasp_outdir}/../results.tsv'
        log_dir = f'{self.grasp_outdir}/../log'
        colnames = self.common_colnames + metric_colnames
        
        super().__init__(ncpu, res_file, log_dir, colnames)

    def run_single(self, pdb_id, ckpt_id, seed, it):
        if self.ana_name == '5jds':
            truth, pred = self.find_file(pdb_id, ckpt_id, seed, it)
            print(truth, pred, sep='\n')
            rmsd, tmscore, dockq_avg = eval(pred, truth)
            return rmsd, tmscore, dockq_avg
        elif self.ana_name == 'simxl':
            truth, pred, restr_file = self.find_file(pdb_id, ckpt_id, seed, it)
            print(truth, pred, restr_file, sep='\n')
            rmsd, tmscore, dockq_avg = eval(pred, truth)
            recall, recall_true = utils.compute_recall(truth, pred, restr_file)
            return rmsd, tmscore, dockq_avg, recall, recall_true
        elif self.ana_name == 'dms':
            truth, pred, fasta = self.find_file(pdb_id, ckpt_id, seed, it)
            print(truth, pred, fasta, sep='\n')
            split = utils.generate_split_ab(fasta)
            rmsd, tmscore, dockq_avg, dockq_dimer = eval(pred, truth, split)
            return rmsd, tmscore, dockq_avg, dockq_dimer
        elif self.ana_name == 'abxl':
            truth, pred, restr_file = self.find_file(pdb_id, ckpt_id, seed, it)
            print(truth, pred, sep='\n')
            split_dict = {
                '1YY9': [[0], [1, 2]],
                '3C09': [[0, 1], [2]],
                '4G3Y': [[0, 1], [2]],
                '6OGE_ABC': [[0], [1, 2]],
                '6OGE_ADE': [[0], [1, 2]],
            }
            split = split_dict[pdb_id.split('_fdr')[0]]
            rmsd, tmscore, dockq_avg, dockq_dimer = eval(pred, truth, split)
            return rmsd, tmscore, dockq_avg, dockq_dimer
        elif self.ana_name == 'cab':
            truth, pred, split_file = self.find_file(pdb_id, ckpt_id, seed, it)
            cids = utils.get_chain_id(truth)
            split = utils.generate_split(split_file=split_file, cids=cids)
            dockq, rmsd = utils.compute_pairwise_dockq_rmsd(pred, truth, split)
            return dockq, rmsd

        

    def find_file(self, pdb_id, ckpt_id, seed, it):
        pred = glob.glob(f'{self.grasp_outdir}/ckpt_{ckpt_id}_{pdb_id}_seed{seed}*_iter{it}.pdb')
        assert len(pred) == 1, pred
        pred = pred[0]

        if self.ana_name == '5jds':
            gt = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/csp/5JDS_processed.pdb'
            return gt, pred
        elif self.ana_name =='simxl':
            pdb_id0, rep_id, fdr, cheat = pdb_id.rsplit('_', 3)
            gt = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/gt_onepdb/{pdb_id0}.pdb'
            restr_file = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/simulate_xl/benchmark_xl_hard/{pdb_id0}_rep{rep_id}.tsv'
            return gt, pred, restr_file
        elif self.ana_name == 'dms':
            pdb_id0 = pdb_id.split('_cheat')[0]
            gt = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/dms_data_full_antigen/data/pdb/{pdb_id0}.pdb'
            fasta = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/dms_data_full_antigen/data/fasta/{pdb_id0}.fasta'
            return gt, pred, fasta
        elif self.ana_name == 'abxl':
            pdb_id0 = pdb_id.split('_fdr')[0]
            gt = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/xl/antibody_crosslink/xl_antibody/pdb/{pdb_id0}.pdb'
            restr_file = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/realdata/xl/antibody_crosslink/xl_antibody/xl/{pdb_id0}.pkl'
            return gt, pred, restr_file
        elif self.ana_name == 'cab':
            pdb_id0, rep_id, dimer = pdb_id.split('_')
            gt = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/colabdock/Antibody_Release/{pdb_id0}/groundtruth/native.pdb'
            split_file = f'/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/colabdock/Antibody_Release/{pdb_id0}/chain_info.txt'
            return gt, pred, split_file
        
            


    def get_args(self):
        def _get_args(file):
            file = os.path.basename(file)
            ckpt_id, pdb_id, seed, it = re.search('ckpt_(\d+)_(.+)_seed(\d+)_score.+_iter(\d+).pdb', file).groups()
            return pdb_id, ckpt_id, seed, it
        all_files = glob.glob(f'{self.grasp_outdir}/*_iter*.pdb')
        args = [_get_args(i) for i in all_files]
        return args

    def eval(self):
        args = self.get_args()
        print(len(args))
        self.run(args)

    def add_info(self):
        df = pd.read_csv(self.res_file, sep='\t')
        print(df.shape)
        dfls = []
        for dff in os.listdir(self.grasp_outdir):
            if not dff.endswith('tsv'):
                continue
            # print(dff)
            ckpt_id, pdb_id, seed = re.search('ckpt_(\d+)_(.+)_seed(\d+)_info', dff).groups()
            # print(ckpt_id, pdb_id, seed)
            df1 = pd.read_csv(f'{self.grasp_outdir}/{dff}', sep='\t')
            df1['ckpt_id'] = int(ckpt_id)
            df1['pdb_id'] = pdb_id
            df1['seed'] = int(seed)
            dfls.append(df1)
        df1 = pd.concat(dfls, axis=0)
        df1 = df1.rename(columns={'Iter':'it'})

        df0 = pd.merge(df, df1.rename(columns={'Iter':'it'}), on=self.common_colnames, how='inner')
        df0.to_csv(f'{self.homedir}/results_all.tsv', sep='\t', index=False)
        print(f'{self.homedir}/results_all.tsv saved')

        df_notconverge = df0.groupby(['pdb_id', 'ckpt_id', 'seed'])['Remove'].min().reset_index()
        df_notconverge = df_notconverge[df_notconverge['Remove']>0]
        df_notconverge = df_notconverge.drop('Remove', axis=1)
        df_notconverge = pd.merge(df0, df_notconverge, on=['pdb_id', 'ckpt_id', 'seed'], how='inner')
        df_notconverge.to_csv(f'{self.homedir}/results_all_notconverge.tsv', sep='\t', index=False)

if __name__ == '__main__':
    rootdir = sys.argv[1] # ft-grasp-v6-notfix-nohard-32_simxl
    ncpu = int(sys.argv[2]) # 64
    ana_name = sys.argv[3] # simxl
    rb = MyRunBatch(ncpu, rootdir, ana_name)
    rb.eval()
    rb.add_info()