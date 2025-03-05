import tempfile
import re
import os
import io
import logging
import subprocess
import sys
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, Chain, Model, PDBIO
from Bio.SeqUtils import seq1
from Bio import Align
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.MMCIFParser import MMCIFParser

class Logger():
    def __init__(self, log_file=None):
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
        if log_file:
            handler = logging.FileHandler(log_file, mode='w')
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        self.logger = logger
    def write(self, message):
        if message != '\n':
            self.logger.info(message.strip())

def scp_file(scpfrom, scpto):
    cmd = f'scp -r {scpfrom} {scpto}'
    subprocess.run(cmd.split(), stderr=-1, stdout=-1, check=True, text=True)

def check_path(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

class RunBatch:
    def __init__(self, ncpu, res_file, log_dir, colnames) -> None:
        self.ncpu = ncpu
        self.res_file = check_path(res_file)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.colnames = colnames

    def _fill_none(self, res_list, max_len=None):
        if max_len is None:
            max_len = max([len(i) for i in res_list])
        # print('maxlen', max_len)
        for i, x in enumerate(res_list):
            res_list[i] = x + (None,) * (max_len-len(x))
        return res_list

    def run_single(self):
        raise NotImplementedError
    
    def run_single_with_logging(self, *args):
        log_file = f'{self.log_dir}/{"_".join([str(i) for i in args])}.log'
        print_logger = Logger(log_file)
        sys.stderr = print_logger
        sys.stdout = print_logger
        try:
            results = self.run_single(*args)
            status = 0
        except Exception as e:
            results = (None,)
            status = 1
            print_logger.logger.error(e, exc_info=True)
        return status, args + results, log_file
    
    def run(self, args_list):
        if os.path.isfile(self.res_file) and len(args_list)>0:
            key_columns = self.colnames[:len(args_list[0])]
            df0 =pd.read_csv(self.res_file, sep='\t')
            df0 = df0[~df0.isna().any(axis=1)]
            df_name = pd.DataFrame(args_list, columns=key_columns)
            for k in key_columns:
                df_name[k] = df_name[k].astype(df0[k].dtype)
            
            df0 = pd.merge(df0, df_name, on=key_columns, how='inner')
            indf0 = df_name.set_index(key_columns).index.isin(df0.set_index(key_columns).index)
            tot_len = len(args_list)
            args_list = [i for i, j in zip(args_list, indf0) if not j]
        else:
            tot_len=len(args_list)
            df0 = None

        print(f'Total {tot_len} cases, remove {tot_len-len(args_list)} cases with results. Run {len(args_list)} cases on {self.ncpu} cpus ...')
        with mp.Pool(self.ncpu) as pool:
            results = pool.starmap(self.run_single_with_logging, args_list)
        status = [i[0] for i in results]
        logfiles = [i[2] for i in results]
        results = [i[1] for i in results]
        results = self._fill_none(results, len(self.colnames))
        print (results[:2])
        df = pd.DataFrame(results, columns=self.colnames)
        if df0 is not None:
            df = pd.concat([df0, df], axis=0)
        df = df.sort_values(by=self.colnames)
        df.to_csv(self.res_file, sep='\t', index=False)
        print(f'Results saved in {self.res_file}')
        errs = [log for log, stat in zip(logfiles, status) if stat]
        if len(errs)>0:
            print(f'{len(errs)}/{len(status)} error(s)!')
            print('Top 5 errors are listed below:')
            print('\n'.join(errs[:5]))


# dms utils
def generate_split_ab(fasta):
    mapping = {}
    with open(fasta, 'r') as f:
        x = [i.strip() for i in f.readlines()][0::2]
    for i in x:
        k = i.split('_')[-1]
        v = i.split('_')[-2]
        mapping[k] = v
    sp_num = list(mapping.keys()).index('sp')
    return [[sp_num], list(set([0,1,2])-set([sp_num]))]


# cab utils

def diff_index(res1, res2):
    if res1 is None:
        return 1
    id1 = res1.id
    id2 = res2.id
    
    if ('CA' in res1) and ('CA' in res2) and (res1['CA'] - res2['CA'] <= 4.0):
        return 1

    if id1[2] == ' ' and id2[2] == ' ':
        return id2[1] - id1[1]
    elif id1[2] == ' ' or id2[2] == ' ':
        return 1
    else:
        return abs(ord(id2[2]) - ord(id2[2]))


def get_full_seq(chain):
    idx = 100000
    previous = None
    seq = ''
    for res in list(chain):
        if res.id[0]!=' ':
            continue
        diff_idx = diff_index(previous, res)
        
        if (diff_idx != 1):
            seq += '-' * (diff_idx - 1)
        seq += seq1(res.resname)
        previous = res.copy()
        idx += diff_idx
        res.id = (' ', idx, ' ')
    return seq


def align_seq(seq_1, seq_2):
    print(f'\n{seq_1}\n{seq_2}\n')
    aligner = Align.PairwiseAligner()
    aligner.target_internal_open_gap_score=-1
    alignments = aligner.align(seq_1, seq_2)  
    best_aln = alignments[0]
    return best_aln

def get_seq(obj):
    if isinstance(obj, str):
        parser = PDBParser(QUIET=True)
        obj = parser.get_structure('none', obj)
    return seq1(''.join([i.resname for i in obj.get_residues()]))

def align_and_clip(pred, truth):
    chain1 = pred
    chain2 = truth
    c1 = Chain.Chain(chain1.id)
    c2 = Chain.Chain(chain2.id)
    s1 = get_seq(chain1)
    s2 = get_full_seq(chain2)
    aln = align_seq(s1, s2)
    assert aln.score == len(s2.replace('-', '')), f'\n{aln}'
    assert len(aln[0]) == len(s1), f'\n{aln}'
    i1 = 0
    i2 = 0
    for i, (a1, a2) in enumerate(zip(*aln)):
        if a1 != '-':
            i1 += 1
        if a2 != '-':
            i2 += 1
        if a1 != '-' and a2 != '-':
            r1 = list(chain1)[i1-1].copy()
            r2 = list(chain2)[i2-1].copy()
            r1.id = (' ', i+1, ' ')
            r2.id = (' ', i+1, ' ')
            c1.add(r1)
            c2.add(r2)
    return c1, c2

def get_ca_pos(obj):
    x = [i['CA'].coord for i in obj.get_residues()]
    return np.array(x)

def compute_rmsd(obj1, obj2):
    cas1, cas2 = get_ca_pos(obj1), get_ca_pos(obj2)
    assert cas1.shape == cas2.shape, f'{cas1.shape} != {cas2.shape}'
    sup = SVDSuperimposer()
    sup.set(cas1, cas2)
    sup.run()
    rmsd = sup.get_rms()
    return rmsd

def merge_chains(cls, cid):
    c = Chain.Chain(cid)
    last = 0
    for chain in cls:
        first=list(chain)[0].id[1]
        for res in list(chain):
            r = res.copy()
            r.id = (r.id[0], r.id[1]+last+1-first, r.id[2])
            c.add(r)
        last=r.id[1]
    return c

def make_two_chains(s, split):
    chain_num = len(list(s.get_chains()))
    m = Model.Model('none')
    if chain_num == 2:
        x = 1
    elif chain_num == len(split[0]+split[1]):
        x = len(split[0])
    else:
        raise ValueError(f'{chain_num} {split}')
    c1 = merge_chains(list(s.get_chains())[:x], 'A')
    c2 = merge_chains(list(s.get_chains())[x:], 'B')
    m.add(c1)
    m.add(c2)
    return m

def align_pdb(pdb_path: str, ref_path: str, output_path, ref_output_path, split):
    parser = PDBParser(QUIET=True)
    pdbio=PDBIO()

    cmd = f'pdb_reres {pdb_path}'
    res = subprocess.run(cmd.split(), stdout=-1, stderr=-1, check=True, text=True).stdout
    f = io.StringIO(res)
    structure1 = parser.get_structure('pred', f)
    structure2 = parser.get_structure('truth', ref_path)
    s1 = make_two_chains(structure1, split)
    s2 = make_two_chains(structure2, split)
    m1 = Model.Model('m1')
    m2 = Model.Model('m2')

    for c1, c2 in zip(s1.get_chains(), s2.get_chains()): 
        
        chain1, chain2 = align_and_clip(c1, c2)
        seq_1 = get_seq(chain1)
        seq_2 = get_seq(chain2)
        print(f'seq len for chain {chain1.id}, {chain2.id}: ({len(c1)}, {len(c2)})===>({len(chain1)}, {len(chain2)})')
        assert seq_1 == seq_2, f'\n{seq_1}\n{seq_2}\n'
        m1.add(chain1)
        m2.add(chain2)
    rmsd = compute_rmsd(m1, m2)        
    pdbio.set_structure(m1)
    pdbio.save(output_path)
    pdbio.set_structure(m2)
    pdbio.save(ref_output_path)
    return rmsd

def generate_split(split_file, cids):
    with open(split_file) as f:
        cont = f.readlines()
    return [[j for j in (re.sub('Chain\d_names: ', '', i.strip())).split(',') if j in cids] for i in cont]

def generate_split_idx(split_file, cids):
    split = generate_split(split_file, cids)
    return [[cids.index(i) for i in j] for j in split]

def get_chain_id(path):
    parser = PDBParser(QUIET=True)
    return [i.id for i in parser.get_structure('none', path).get_chains()]

def compute_pairwise_dockq_rmsd(pred, truth, split, save=None):
    dockq_path = '/lustre/grp/gyqlab/share/xieyh/EvalComplex/dockq/DockQ.py'
    # p_cids = get_chain_id(pred)
    with tempfile.NamedTemporaryFile('w') as temp_file1,\
        tempfile.NamedTemporaryFile('w') as temp_file2:
        tmp1 = temp_file1.name
        tmp2 = temp_file2.name
        if save:
            tmp1 = check_path(f'{save}_pred.pdb')
            tmp2 = check_path(f'{save}_truth.pdb')
        rmsd = align_pdb(pred, truth, tmp1, tmp2, split)
        # print(f'TMscore -c -ter 0 {pred} {gt}')
        s1 = get_seq(tmp1)
        s2 = get_seq(tmp2)
        assert s1 == s2, f'\n{s1}\n{s2}\n'
        cmd = f'{dockq_path} {tmp1} {tmp2} -no_needle'
        print(cmd)
        res = subprocess.run(cmd.split(), stderr=-1, stdout=-1, check=True, text=True).stdout
        print(res)
        dockq = re.search('\nDockQ (.*)\n', res).group(1)
    return float(dockq), rmsd


# dimer mode remove gap
def remove_gap(pdb_path):

    parser = PDBParser(QUIET=True)
    # pdbio=PDBIO()
    s = parser.get_structure('none', pdb_path)
    cids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    model = Model.Model('none')
    ci = 0
    cur_chain = None
    for c in s.get_chains():
        if cur_chain is not None:
            model.add(cur_chain)


        cur_chain = Chain.Chain(cids[ci])
        ci += 1
        last_id = None
        for r in c.get_residues():
            cur_id = r.id[1]
            if (last_id is None) or (cur_id != last_id+200):
                cur_chain.add(r.copy())
                last_id = cur_id
            else:
                # new chain
                model.add(cur_chain)
                cur_chain = Chain.Chain(cids[ci])
                ci += 1
                cur_chain.add(r.copy())
                last_id = cur_id


    model.add(cur_chain)
    return model

##########################################################################################
### used to split chains from the predictions of ColabDock ================================
##########################################################################################
def split_chains(fasta, pdb, out_pdb=None):
    print('Splitting chains in pdb file based on fasta file')
    print('fasta:', fasta)
    print('pdb:', pdb)
    print('out_pdb:', out_pdb, flush=True)
    parser = PDBParser(QUIET=True)
    if pdb.endswith('cif'):
        parser = MMCIFParser(QUIET=True)
    io = PDBIO()

    with open(fasta, 'r') as f:
        seqs = [i.strip().replace('X', '') for i in f.readlines()[1::2]]

    cids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

    s = parser.get_structure('X', pdb)
    num_chains = len(list(s.get_chains()))
    if num_chains == len(seqs):
        m=s[0]
    else:
        residues = list(s.get_residues())
        seq_pdb = ''.join([seq1(r.resname) for r in s.get_residues()])
        
        def reorder_seqs(seqs, tot_seq):
            new_seqs = []
            t=0
            while tot_seq:
                for i in seqs:
                    print(i, tot_seq)
                    if tot_seq.startswith(i):
                        new_seqs.append(i)
                        tot_seq = tot_seq[len(i):]
                        seqs.remove(i)
                        break
                t+=1
                if t>100:
                    print('Error: cannot find the right order of the sequences')
                    raise ValueError
            return new_seqs
        print(seqs, seq_pdb)
        seqs = reorder_seqs(seqs, seq_pdb)

        seq_fasta = ''.join(seqs)
        assert seq_pdb == seq_fasta, f'\n{seq_pdb}\n{seq_fasta}\n'
        seq_lens = [len(i) for i in seqs]
        start_idx = 0
        m = Model.Model(0)
        for cid, l in zip(cids, seq_lens):
            chain = Chain.Chain(cid)
            for i in range(start_idx, start_idx+l):
                chain.add(residues[i].copy())
            
            start_idx += l
            m.add(chain)

    if out_pdb is None:
        return m
    else:
        io.set_structure(m)
        io.save(out_pdb)

##########################################################################################
### used to compute recall =============================================================
##########################################################################################
BINS = np.arange(4, 33, 1)
def reorder_seq_dict(seq_dict):
    # this function should return a new dictionary that maps the chain ids to the sequences in the order of the restraints dict,
    # where same sequences are grouped together
    # for example, if the restraints dict is {'A': 'ACGT', 'B': 'CGTA', 'C': 'ACGT', 'D': 'CGTA'}, the function should return:
    # {'A': 'ACGT', 'C': 'ACGT', 'B': 'CGTA', 'D': 'CGTA'}
    
    # get all unique values in the sequence dictionary
    unique_values_order = []
    for value in seq_dict.values():
        if value not in unique_values_order:
            unique_values_order.append(value)
    
    # create a new dictionary with the same order as the restraints dict
    reordered_dict = {k: v for v in unique_values_order for k in seq_dict if seq_dict[k] == v}
    
    return reordered_dict


def get_fasta_dict(fasta_file):
    # this function should return a dictionary that maps fasta chain ids to its sequence, 
    # for example, if the fasta file contains two sequences, the dictionary should be:
    # {'A': 'ACGT', 'B: 'CGTA'}
    with open(fasta_file, 'r') as f:
        fasta_dict = {}
        seq = ''
        for line in f.readlines():
            if line.startswith('>'):
                if seq:
                    fasta_dict[desc] = seq
                seq = ''
                desc = line[1:].strip()
                assert desc not in fasta_dict, f'Duplicate chain description {desc} in fasta file'
            else:
                seq += line.strip()
        if seq:
            fasta_dict[desc] = seq
    return fasta_dict

def get_asym_id(fasta_dict):
    # this function should return the asym_id of the fasta_dict
    ids = [np.repeat(i+1, len(seq)) for i, seq in enumerate(fasta_dict.values())]
    return np.concatenate(ids)

def get_pseudo_beta(residues_list):
    # this function should return the pseudo-beta coordinates of the residues_list and mask
    # the None in the list should be replaced by the (0,0,0) and masked in mask

    pseudo_beta_coords = []
    mask = []
    for residue in residues_list:
        if residue is None:
            pseudo_beta_coords.append((0,0,0))
            mask.append(0)
        else:
            if 'CB' in residue:
                pseudo_beta_coords.append(residue['CB'].get_coord())
                mask.append(1)
            elif 'CA' in residue:
                pseudo_beta_coords.append(residue['CA'].get_coord())
                mask.append(1)
            else:
                pseudo_beta_coords.append((0,0,0))
                mask.append(0)
    return np.array(pseudo_beta_coords), np.array(mask)

def get_seq_from_chain(chain):
    # this function should return the sequence of a chain object
    # the gap in sequence should be represented by dot ('.')
    # for example, if the chain contains two residues, 
    # the function should return 'A..T'
    seq = ''
    last_idx = 0
    for residue in chain.get_residues():
        assert residue.id[0] == ' ', residue
        if last_idx is not None and residue.id[1] != last_idx + 1:
            seq += '.' * (residue.id[1] - last_idx - 1)
        seq += seq1(residue.resname)
        last_idx = residue.id[1]
    
    return seq

def find_chain(seq, model, exlude_chains=None):
    # this function should return the list of residue objects 
    # in the chain object of the model that matches the sequence
    # the gap in the sequence should be represented by None in the list.
    # the chain id should also be returned. 
    # exlude_chains is a list of chain ids that should be excluded from the search
    # for example, if the sequence is 'ACGT' and the model contains two chains, 
    # the function should return:
    # [residue_A, residue_C, None, None], chain_id
    for chain in model.get_chains():
        if exlude_chains and chain.get_id() in exlude_chains:
            continue
        seq_chain = get_seq_from_chain(chain)
        match = re.match(seq_chain, seq)
        if match:
            span = match.span()
            seq_chain_full = '.'*span[0] + seq_chain + '.'*(len(seq)-span[1])
            residues = []
            i = 0
            for c in seq_chain_full:
                if c == '.':
                    residues.append(None)
                else:
                    residues.append(list(chain.get_residues())[i])
                    i += 1
            return residues, chain.id
        
    return None, None

def get_residues_from_fasta_dict(model, fasta_dict):
    # this function should return the list of residue objects 
    # in the model that matches the sequences in the fasta_dict
    # the gap in the sequence should be represented by None in the list.

    exclued_chains = []
    residues_list = []
    print('find chains in model')
    for seq in fasta_dict.values():
        print(exclued_chains, seq)
        residues, chain_id = find_chain(seq, model, exlude_chains=exclued_chains)
        print('find_chain result:', chain_id)
        print('Residues chain seq:', ''.join([seq1(r.resname) if r is not None else '.' for r in residues]))
        assert chain_id is not None, f'Cannot find chain {seq} in model'
        exclued_chains.append(chain_id)
        residues_list.extend(residues)
    print('Residues list seq:', ''.join([seq1(r.resname) if r is not None else '.' for r in residues_list]))
    return residues_list


def parse_restraints(restraints):
    # this function should parse the restraints dict and return a list of restraints
    # each restraint should be a tuple of (residue1, residue2, distance) or (residue1, distance)
    # for example, 
    # the function may return:
    # [(residue_A, residue_B, 3.5), (residue_C, residue_D, 4.0), (residue_E, 5.0)]
    restraints_list = []

    # interface restraints
    # print(restraints)
    for i in np.where(restraints['interface_mask'])[0]:
        restraints_list.append((i, 8.0))

    # sbr restraints
    for i, j in zip(*np.where(restraints['sbr_mask'])):
        if i>j:
            continue
        distri = restraints['sbr'][i, j]
        cutoff_idx = max(np.where(distri > 1/distri.size)[0])
        # print(cutoff_idx)
        cutoff = np.concatenate([BINS, [np.inf]])[cutoff_idx]
        restraints_list.append((i, j, cutoff))

    return restraints_list

def check_single_restraint_status(dist_mat, mask, asym_id, restraint):
    # this function should check the status of a single restraint
    # the restraint should be a tuple of (residue1, residue2, distance) or (residue1, distance)
    # the function should return True if the restraint is satisfied, False if the restraint is unsatisfied, and None if the restraint is incorrect
    if len(restraint) == 2:
        i, dist0 = restraint
        interface_dist = (dist_mat + (asym_id[None] == asym_id[:, None])*10000).min(axis=0) + (1-mask)*10000
        dist = interface_dist[i]
        satis =  dist <= dist0
        print(f'IR Restraint {i}, dist: {dist:.4f}, cutoff: {dist0:.4f}, satis: {satis}')
    else:
        i, j, dist0 = restraint
        # print(dist_mat.shape, mask.shape, asym_id.shape, i, j, dist0)
        dist = dist_mat[i, j] + (1-mask[i]*mask[j])*10000
        satis = dist <= dist0
        print(f'SBR Restraint {i}, {j}, dist: {dist:.4f}, cutoff: {dist0:.4f}, satis: {satis}')
    return satis, dist


def compute_recall(mp, mt, fasta, restr, return_num=False):
    fasta_dict = reorder_seq_dict(get_fasta_dict(fasta))
    print('fasta_dict:', fasta_dict)


    with open(restr, 'rb') as f:
        restraints = pickle.load(f)

    asym_id = get_asym_id(fasta_dict)
    residues_gt = get_residues_from_fasta_dict(mt, fasta_dict)
    residues_pred = get_residues_from_fasta_dict(mp, fasta_dict)
    restraints_list = parse_restraints(restraints)


    pseudo_beta_coords_gt, mask_gt = get_pseudo_beta(residues_gt)
    dist_mat_gt = np.sqrt(np.sum((pseudo_beta_coords_gt[None] - pseudo_beta_coords_gt[:, None])**2, axis=-1))

    pseudo_beta_coords_pred, mask_pred = get_pseudo_beta(residues_pred)
    dist_mat_pred = np.sqrt(np.sum((pseudo_beta_coords_pred[None] - pseudo_beta_coords_pred[:, None])**2, axis=-1))

    satis_num = 0
    tot_num = 0
    satis_correct_num = 0
    correct_num = 0
    for restraint in restraints_list:
        print('GROUND TRUTH')
        satis_gt, dist_gt = check_single_restraint_status(dist_mat_gt, mask_gt, asym_id, restraint)
        print('PREDICTION')
        satis_pred, dist_pred = check_single_restraint_status(dist_mat_pred, mask_pred, asym_id, restraint)
        
        # for computing the recall
        if satis_pred:
            satis_num += 1
        tot_num += 1

        if satis_gt:
            correct_num += 1
            if satis_pred:
                satis_correct_num += 1
    recall = satis_num / tot_num
    recall_true = satis_correct_num / correct_num
    print(f'Satisfied restraints: {satis_num}, Total restraints: {tot_num}, Recall: {recall:.4f}')
    print(f'Correct restraints: {correct_num}, Correctly satisfied restraints: {satis_correct_num}, Recall_true: {recall_true:.4f}')
    if return_num:
        return recall, recall_true, {'satis_num': satis_num, 'tot_num': tot_num, 'satis_correct_num': satis_correct_num, 'correct_num': correct_num}
    return recall, recall_true

if __name__ == '__main__':
    pred = '/lustre/grp/gyqlab/xieyh/multimer/test_dataset/20230320/grasp_infer_res/results/ft-grasp-v11-64_colabdock_ab/data/ckpt_20000_3HMX_1_dimer1_seed32981_score16.84_iter2.pdb'
    model = remove_gap(pred)
    pdbio = PDBIO()
    pdbio.set_structure(model)
    pdbio.save('test.pdb')
