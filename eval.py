
import re
import os
import warnings
# import argparse
import tempfile
import shutil
import subprocess
import pandas as pd
import numpy as np
import multiprocessing as mp

from utils import compute_recall
from Bio.Align import PairwiseAligner
from Bio import BiopythonWarning
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from dockq.DockQ import calc_DockQ
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', BiopythonWarning)


THREE_TO_ONE ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y', \
    'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A', \
    'GLY':'G', 'PRO':'P', 'CYS':'C', 'UNK': 'X', 'SEC': 'U', 'PYL': 'O'} # SEC 硒代半胱氨酸， PYL 吡咯赖氨酸

def get_seq(obj):
    seq = ''
    for i in obj.get_residues():
        if 'CA' in i:
            seq += THREE_TO_ONE.get(i.resname, 'X')
        else:
            seq += 'X'
    return seq


def diff_index(res1, res2):
    if res1 is None:
        return 1
    id1 = res1.id
    id2 = res2.id
    

    if ('CA' in res1) and ('CA' in res2) and (res1['CA'] - res2['CA'] <= 4.0) and (id2[1] - id1[1] > 20):
        print('warning, there is a large gap in index with close dist')
        return 1

    if id1[2] == ' ' and id2[2] == ' ':
        # print('case1', id1, id2)
        return id2[1] - id1[1]
    elif (id1[2] == ' ') or (id2[2] == ' '):
        # print('case2',  id1, id2)
        return 1
    else:
        # print('case3',  id1, id2)
        return abs(ord(id2[2]) - ord(id1[2]))


def each_chain_start_from_one(path):
    if isinstance(path, str):
        if path.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        elif path.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
        s = parser.get_structure('none', path)
    else:
        s = path
    new_model = Model(0)
    for chain in s.get_chains():
        new_chain = Chain(chain.id)
        last_r = None
        idx = 0
        for r in chain.get_residues():
            new_r = r.copy()
            delta = diff_index(last_r, r)
            # if delta >1:
            #     print('Gap occured')
            idx += delta
            new_r.id = (' ', idx, ' ')
            new_chain.add(new_r)
            last_r = r
        new_model.add(new_chain)
    return new_model 

def pad_zero_for_first_axis(arr, pad_before, pad_after):
    pad_width = np.zeros((len(arr.shape), 2))
    pad_width[0, :] = (pad_before, pad_after)
    return np.pad(arr, pad_width.astype(np.int32))


# def check_match(pred_seq, truth_seq):
#     # print(truth_seq.replace('X', '.'), pred_seq)
#     return re.search(truth_seq.replace('X', '.'), pred_seq)

# def get_mask(pred_seq, truth_seq):
#     mask = np.array([i!='X' for i in truth_seq], dtype=np.int32)
#     # print(f'mask: {mask}')
#     mask =  insert_gaps_at_two_ends(pred_seq, truth_seq, mask)
#     return mask.astype(bool)


def check_match_and_get_mask(pred_seq, truth_seq, max_mismatch=3, match_dict={}):

    key = f'{pred_seq}+{truth_seq}'
    if key in match_dict:
        # print('pass')
        return match_dict[key]

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.wildcard = 'X'
    aligner.match_score = 0
    aligner.mismatch_score = -1
    aligner.target_gap_score = -100

    aligner.query_internal_gap_score = -100
    aligner.query_end_gap_score = 0

    alignments = aligner.align(pred_seq, truth_seq)
    best_alignment = alignments[0]

    is_match = best_alignment.score >= (-max_mismatch)
    
    if is_match:
        if best_alignment.score<0:
            print(f'Warning: There are {-int(best_alignment.score)} mismatch between target and query')
            print(best_alignment)
        x = np.where([i!='-' for i in best_alignment[1]])[0]
        pad_width = (x.min(), len(pred_seq)-x.max()-1)
        mask = np.array([(i==j) and (j!='X') for i, j in zip(*best_alignment)], dtype=bool)
    else:
        pad_width=None
        mask=None

    match_dict[key] = [is_match, pad_width, mask]
    return match_dict[key]

def insert_gaps_at_two_ends(arr, pad_width):
    pad_before, pad_after = pad_width
    return pad_zero_for_first_axis(arr, pad_before, pad_after)


def get_ca_and_mask(ca_t, truth_seq, pred_seq, match_dict, max_mismatch):
    assert ca_t.shape[0] == len(truth_seq)
    is_match, pad_width, mask = check_match_and_get_mask(pred_seq, truth_seq, match_dict=match_dict, max_mismatch=max_mismatch)
    # print(is_match, pad_width, mask.shape, mask.sum())
    ca_t = insert_gaps_at_two_ends(ca_t, pad_width)
    return ca_t, mask



def parse_chain(chain):
    seq = ''
    cals = []
    last_res_idx = 0
    gap_positions = []
    for res in chain.get_residues():
        
        if res.id[0] != ' ':
            continue
        # print(res, res.id)
        res_idx = int(res.id[1])
        
        delta_idx = res_idx - last_res_idx
        gap_positions.extend(list(range(last_res_idx + 1, res_idx)))
        if delta_idx != 1:
            seq += 'X' * (delta_idx - 1)
            for _ in range(delta_idx - 1):
                cals.append([0, 0, 0])
        last_res_idx = res_idx
        
        resname = THREE_TO_ONE[res.resname] if 'CA' in res else 'X'
        seq += resname
        if resname == 'X':
            cals.append([0, 0, 0])
        else:
            for at in res.get_atoms():
                if at.id == 'CA':
                    cals.append(at.coord)
                

    ca_pos = np.array(cals)
    # print(ca_pos.shape)
    assert len(seq) == ca_pos.shape[0], (len(seq), ca_pos.shape)
    mask = np.array([i != 'X' for i in seq], dtype=bool)

    # insert UNK residues at gap positions
    for gap_position in gap_positions:
        unk_residue = Residue((' ', gap_position, ' '), 'UNK', 0)
        chain.insert(gap_position-1, unk_residue) # pos is 0-indexed
    # print([i.id[1] for i in chain.get_residues()])
    assert get_seq(chain) == seq, f'\n{get_seq(chain)}\n{seq}\n'
    print(chain.id, seq)
    return seq, ca_pos , mask


def cal_rmsd(x1, x2, eps = 1e-6):
    assert x1.shape == x2.shape, (x1.shape, x2.shape)
    assert x1.shape[-1] == 3
    return np.sqrt(((x1 - x2) ** 2).sum(-1).mean() + eps)

def kabsch_rmsd(true_atom_pos, pred_atom_pos):
    r, x = get_optimal_transform(
        true_atom_pos,
        pred_atom_pos
    )
    aligned_true_atom_pos = true_atom_pos @ r + x
    return cal_rmsd(aligned_true_atom_pos, pred_atom_pos)


def cal_ca_kabsch_rmsd(pred_ca, truth_ca, truth_cids, pm, match_dict,max_mismatch):
    # truth_ca[chain_id] = [ca_pos, mask, heav_pos]
    # pred_ca[chain.id] = ca_pos
    pred_ca_ls = []
    truth_ca_ls = []
    for truth_cid, pred_idx in zip(truth_cids, pm):
        truth_ca_pos, truth_seq, truth_mask =truth_ca[truth_cid]
        pred_ca_pos, pred_seq = list(pred_ca.values())[pred_idx]
        truth_ca_pos, mask = get_ca_and_mask(truth_ca_pos, truth_seq, pred_seq, match_dict=match_dict, max_mismatch=max_mismatch)
        truth_ca_ls.append(truth_ca_pos[mask])
        pred_ca_ls.append(pred_ca_pos[mask])
        truth_ca_all = np.concatenate(truth_ca_ls)
        pred_ca_all = np.concatenate(pred_ca_ls)
    return kabsch_rmsd(truth_ca_all, pred_ca_all)

def get_optimal_transform(src_atoms, tgt_atoms, mask = None):    
    assert src_atoms.shape == tgt_atoms.shape, (src_atoms.shape, tgt_atoms.shape)
    assert src_atoms.shape[-1] == 3
    if mask is not None:
        if mask.dtype != bool:
            mask = mask.astype(bool)
        assert mask.shape[-1] == src_atoms.shape[-2]
        if mask.sum() == 0:
            src_atoms = np.zeros((1, 3)).astype(np.float32)
            tgt_atoms = src_atoms
        else:
            src_atoms = src_atoms[mask, :]
            tgt_atoms = tgt_atoms[mask, :]
    src_center = src_atoms.mean(-2, keepdims=True)
    tgt_center = tgt_atoms.mean(-2, keepdims=True)

    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x




def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = P.transpose(-1, -2) @ Q
    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = V @ W
    return U




def get_mean_pred(pred_ca_pos, pred_seq, truth_seq, match_dict, max_mismatch):
    assert pred_ca_pos.shape[-1] == 3
    
    is_match, _, mask = check_match_and_get_mask(pred_seq, truth_seq, match_dict=match_dict, max_mismatch=max_mismatch)

    if not is_match:
        return np.array([1e9, 1e9, 1e9]).reshape(1, -1)
    else:
        # print(pred_ca_pos, mask)
        return pred_ca_pos[mask].mean(0, keepdims=True)


def find_optimal_permutation(x_mean_pred, x_mean_truth):
    '''Find Optimal Permutation'''
    assert x_mean_pred.shape[-1] == 3, x_mean_pred.shape
    assert x_mean_truth.shape[-1] == 3, x_mean_truth.shape
    d_kl = np.sqrt(((x_mean_pred - x_mean_truth[None]) ** 2).sum(-1))
    p_l = []
    for l in range(d_kl.shape[1]):
        d_l = d_kl[:, l]
        # print(d_l)
        best_idx = d_l.argmin()
        # print(best_idx)
        p_l.append(best_idx)
        d_kl[best_idx, :] = 1e9
    return p_l

def get_heav_pos(chain):
    heavs = []
    for a in chain.get_atoms():
        if a.element != 'H':
            heavs.append(a.coord)
    return np.array(heavs)


def has_contact(chain1, chain2):
    '''defined as any heavy atom of one chain being within 6A of any heavy atom of the other chain'''
    heavpos1 = get_heav_pos(chain1)
    heavpos2 = get_heav_pos(chain2)
    d = np.expand_dims(heavpos1, 1) - heavpos2[None, :]
    dist = np.sqrt((d ** 2).sum(-1))
    # print('dist', dist.min())
    return (dist <= 6).any()


def check_match_table(df, match_dict):
    rm_ls = []
    for tcid in df.truth_cid.unique():
        if df.truth_cid.value_counts()[tcid] > 1:
            print(f'Warning: truth chain {tcid} has multiple predictions')
            pcids = df.pred_cid[df.truth_cid == tcid].values
            for pcid in pcids:
                tcids = df.truth_cid[df.pred_cid == pcid].values
                for other_pcid in pcids:
                    if other_pcid == pcid:
                        continue
                    tcids_other = df.truth_cid[df.pred_cid == other_pcid].values
                    if len(tcids_other) > len(tcids):
                        print(f'Warning: remove match pair pcid {other_pcid} and tcid {tcid} from match table')
                        rm_ls.append(f'{other_pcid}+{tcid}')
    rm_ls = list(set(rm_ls))
    print('rm_ls:', rm_ls)
    drop_lines = []
    if len(rm_ls) > 0:
        for i, row in df.iterrows():
            if f'{row.pred_cid}+{row.truth_cid}' in rm_ls:
                match_dict[f'{row.pred_seq}+{row.truth_seq}'][0] = False
                drop_lines.append(i)
        df = df.drop(index=drop_lines)
        print(f'After removing {len(rm_ls)} pairs, the match table is:\n')
        print(df[['pred_cid', 'seq_len', 'num_chains', 'truth_cid', 'true_seq_len']].to_string())
    return df



def rm_masked_res(chain, mask, idx_shift=0):
    chain1 = Chain(chain.id)
    assert len(chain.child_list) == len(mask), (len(chain.child_list), len(mask))
    for i, (res, m) in enumerate(zip(chain.child_list, mask)):
        if m:
            res1 = res.copy()
            res1.id = (' ', i+idx_shift+1, ' ')
            chain1.add(res1)
    return chain1



def align_pred_to_truth(pred_pdb, truth_pdb, max_mismatch=3):
    # pred_pdb: str or Model
    # truth_pdb: str or Model or list of chains
    # return: pred_model, truth_model, rmsd_min
    # chain order is preserved to match the order of truth pdb
    # residue index starts from 1 and is preserved to match the order of pred pdb
    # the masked residues are removed from the pred_model and truth_model

    pred = each_chain_start_from_one(pred_pdb)
    truth_pdbs = list(each_chain_start_from_one(truth_pdb).get_chains())

    chain_dict = {}
    pred_ca = {}
    print('Start parse prediction chains ... ')
    for chain in pred.get_chains():
        seq, ca_pos, _ = parse_chain(chain)
        pred_ca[chain.id] = (ca_pos, seq)

        if seq in chain_dict:
            chain_dict[seq].append(chain.id)
        else:
            chain_dict[seq] = [chain.id]
        
    ls = []
    for k, v in chain_dict.items():
        ls.append([''.join(v), len(k), k])
    df_pred = pd.DataFrame(ls, columns=['pred_cid', 'seq_len', 'pred_seq'])
    df_pred['num_chains'] = df_pred.pred_cid.map(len)

    # ground truth pdb
    ls = []
    truth_ca = {}
    truth_chain = {}
    
    ori_order = [c.id for c in truth_pdbs]
   
    
    match_dict = {}

    print('Start parse truth chains ... ')
    for chain in truth_pdbs:
        truth_chain[chain.id] = chain
        # print('truth', chain.id)
        seq, ca_pos, mask = parse_chain(chain)
        truth_ca[chain.id] = [ca_pos, seq, mask]

        for _, row in df_pred.iterrows():
            if row.seq_len < len(seq):
                continue
            if check_match_and_get_mask(row.pred_seq, seq, match_dict=match_dict, max_mismatch=max_mismatch)[0]:
                ls.append([*row, chain.id, seq, mask])
                # print(f'ls append truth {chain.id}')
            # else:
            #     print(row.pred_seq, seq, sep='\n')
    assert len(truth_pdbs) == len(pred.child_list), f'The number of ground truth chains is not equal to that of prediction: {len(truth_pdbs), len(pred.child_list)}'
            
            
    df = pd.DataFrame(ls, columns=df_pred.columns.tolist() + ['truth_cid', 'truth_seq', 'mask']) 
    df['true_seq_len'] = df['mask'].map(np.sum)
    df = df.sort_values(by=['num_chains', 'true_seq_len'], ascending=[True, False])
    print(df[['pred_cid', 'seq_len', 'num_chains', 'truth_cid', 'true_seq_len']].to_string())
    print('check match table ...')
    df = check_match_table(df, match_dict)
    # print(df.shape)
    # print(df.truth_cid.value_counts())
    truth_cids = df.truth_cid.unique()
    # print('truth chains:', truth_cids, len(truth_cids))
    # print('pred chains', pred_ca.keys(), len(pred_ca.keys()))
    # assert len(pred_ca) == len(truth_cids)

    anchor_truth = truth_cids[0]
    anchors_pred = list(''.join(df.pred_cid[df.truth_cid == anchor_truth]))#list(df.pred_cid[0])
    anchors_pred = list(set(anchors_pred))
    anchors_pred.sort()
    print(f'anchor_truth: {anchor_truth}; anchors_pred: {anchors_pred}')

    truth_seqs = [truth_ca[i][1] for i in truth_cids]
    # x_mean_pred: (num_pred_chain, num_truth_chain, 3)
    x_mean_pred = np.concatenate(
        [np.concatenate([get_mean_pred(pred_ca_pos, pred_seq, truth_seq, match_dict=match_dict, max_mismatch=max_mismatch) 
            for pred_ca_pos, pred_seq in pred_ca.values()])[:, None] 
        for truth_seq in truth_seqs], 1)

    pm_best = []
    rmsd_min = 1e9
    for anchor_pred in anchors_pred:
        ca_t, truth_seq, _ = truth_ca[anchor_truth]
        ca_p, pred_seq = pred_ca[anchor_pred]
        ca_t, mask = get_ca_and_mask(ca_t, truth_seq, pred_seq, match_dict, max_mismatch=max_mismatch)
        r, t = get_optimal_transform(ca_t, ca_p, mask)
        x_mean_truth = np.concatenate([(truth_ca[i][0][truth_ca[i][2]] @ r + t).mean(0, keepdims=True) for i in truth_cids])
        pm = find_optimal_permutation(x_mean_pred, x_mean_truth)
        rmsd = cal_ca_kabsch_rmsd(pred_ca, truth_ca, truth_cids, pm, match_dict=match_dict, max_mismatch=max_mismatch)
        print([anchor_truth, anchor_pred, pm, rmsd])
        if rmsd < rmsd_min:
            rmsd_min = rmsd
            pm_best = pm

    # generate models for pred and truth
    match_table = {}
    for cid_t, cid_p in zip(truth_cids, np.array(list(pred_ca.keys()))[pm_best]):
        # cids_p = df.pred_cid[df.truth_cid == cid_t].values[0]
        # assert cid_p in cids_p, (cid_p, cids_p)
        match_table[cid_t] = cid_p
    print(rmsd_min, match_table)
    model_p = Model(0)
    model_t = Model(1)
    print(ori_order)
    for cid_t in ori_order:
        cid_p = match_table[cid_t]
        ca_t, truth_seq, mask_truth = truth_ca[cid_t]
        ca_p, pred_seq = pred_ca[cid_p]

        is_match, (pad_before, pad_after), mask_pred = check_match_and_get_mask(pred_seq, truth_seq, match_dict=match_dict)
        
        mask_truth = mask_pred[pad_before:(len(mask_pred)-pad_after)]
        model_p.add(rm_masked_res(pred.child_dict[cid_p], mask_pred))
        model_t.add(rm_masked_res(truth_chain[cid_t], mask_truth, idx_shift=pad_before))
        
    for r_p, r_t in zip(model_p.get_residues(), model_t.get_residues()):
        assert r_p.id[1] == r_t.id[1], (r_p.id, r_t.id)
        assert r_p.resname == r_t.resname, (r_p.resname, r_t.resname)
    # assert get_seq(model_t) == get_seq(model_p), f'\n{get_seq(model_t)}\n{get_seq(model_p)}\n'
    return model_p, model_t, rmsd_min


def make_one_chain(chainls, cid):
    new_chain = Chain(cid)
    i = 1
    for chain in chainls:
        for res in chain:
            new_res = res.copy()
            new_res.id = (' ', i, ' ')
            i += 1
            new_chain.add(new_res)
    return new_chain


def make_two_chain_pdb(chainls, lsls, file):
    if not isinstance(chainls, list):
        chainls = list(chainls.get_chains())
    io = PDBIO()
    m = Model(0)
    for cid, ls in zip('AB', lsls):
        m.add(make_one_chain([chainls[i] for i in ls], cid))         
    io.set_structure(m)
    io.save(file)


def cal_dockq(pred_list, truth_list, lsls):
    with tempfile.NamedTemporaryFile('w') as fp,\
        tempfile.NamedTemporaryFile('w') as ft:
        make_two_chain_pdb(pred_list, lsls, fp.name)
        make_two_chain_pdb(truth_list, lsls, ft.name)
        # make_two_chain_pdb(pred_list, lsls, '/lustre/grp/gyqlab/share/xieyh/pred_tmp.pdb')
        # make_two_chain_pdb(truth_list, lsls, '/lustre/grp/gyqlab/share/xieyh/truth_tmp.pdb')
        # try:
        info = calc_DockQ(fp.name, ft.name)
        dockq = info['DockQ']
        # except:
        #     print('DockQ calculation failed')
        #     shutil.copyfile(fp.name, './pred_tmp.pdb')
        #     shutil.copyfile(ft.name, './truth_tmp.pdb')
        #     print('pred_tmp.pdb and truth_tmp.pdb saved in current directory')
        #     print('DockQ calculation failed, Please check the input pdb files and run DockQ calculation manually, DockQ was set to 0')
        #     dockq = 0
    return dockq


def cal_dockq_avg(pred, truth):
    # print('cal_dockq_avg')
    truth_list = list(truth.get_chains())
    pred_list = list(pred.get_chains())
    n_chains = len(truth_list)
    dockqls = []
    for i in range(n_chains - 1):
        for j in range(i + 1, n_chains):
            cont = has_contact(truth_list[i],truth_list[j])
            # print(i, j, cont)
            if not cont:
                continue
            dockq = cal_dockq(pred_list, truth_list, [[i], [j]])
            dockqls.append(dockq)
    if len(dockqls) == 0:
        print('No contact chain-pair found')
    return np.mean(dockqls)

def cal_tmscore(pred, truth):
    io = PDBIO()
    with tempfile.NamedTemporaryFile('w') as fp,\
        tempfile.NamedTemporaryFile('w') as ft:
        io.set_structure(pred)
        io.save(fp.name)
        io.set_structure(truth)
        io.save(ft.name)
        cmd = f'MMalign {fp.name} {ft.name}'
        res = subprocess.run(cmd.split(), stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True, text=True)
        res = res.stdout
        tmscore = re.search('TM-score= (.*) \(', res)
        if tmscore:
            tmscore = float(tmscore.group(1))
        else:
            tmscore = None
        print(res)
        return tmscore


def eval(pred, truth, split=None, fasta=None, restr=None, max_mismatch=3):
    res = []
    mp, mt, rmsd = align_pred_to_truth(pred, truth, max_mismatch=max_mismatch)
    res.append(rmsd)
    tmscore = cal_tmscore(mp, mt)
    res.append(tmscore)
    dockq_avg = cal_dockq_avg(mp, mt)
    res.append(dockq_avg)
    if split is not None:
        dockq_dimer = cal_dockq(mp, mt, split)
        res.append(dockq_dimer)
    if fasta is not None and restr is not None:
        recall, recall_true = compute_recall(mp, mt, fasta, restr)
        res.append(recall)
        res.append(recall_true)
    return [round(i, 3) if i is not None else i for i in res]



def run_cmd(cmd):
    res = subprocess.run(cmd.split(), capture_output=True, text=True)
    if res.returncode == 0:
        return res.stdout
    else:
        print(res.stdout)
        print(res.stderr)
        raise ValueError(f'Command {cmd} failed')
def rm_oxt(pdb):
    for r in pdb.get_residues():
        for a in r.get_atoms():
            if a.id == 'OXT':
                r.detach_child(a.id)
        
    return pdb
def cal_lddt(pred, truth, stereo_file='/lustre/grp/gyqlab/xieyh/app/lddt/stereo_chemical_props.txt'):
    # usage: lddt [options] <mod1> [mod1 [mod2]] <re1>[,ref2,ref3]
    #    -s         selection performed on ref
    #    -c         use Calphas only
    #    -f         perform structural checks and filter input data
    #    -t         fault tolerant parsing
    #    -p <file>  use specified parmeter file. Mandatory
    #    -v <level> verbosity level (0=results only,1=problems reported, 2=full report)
    #    -b <value> tolerance in stddevs for bonds
    #    -a <value> tolerance in stddevs for angles
    #    -r <value> distance inclusion radius
    #    -i <value> sequence separation
    #    -e         print version
    #    -x         ignore residue name consistency checks
    io = PDBIO()
    pred = rm_oxt(pred)
    truth = rm_oxt(truth)
    with tempfile.NamedTemporaryFile('w') as fp, \
        tempfile.NamedTemporaryFile('w') as ft:
        io.set_structure(pred)
        io.save(fp.name)
        io.set_structure(truth)
        io.save(ft.name)

        cmd = f'lddt -f -p {stereo_file} -b 15 -a 15 -r 15 -i 0 {fp.name} {ft.name}'
        res = run_cmd(cmd)
        print(res)
        lddt = float(re.search('Global LDDT score:(.+)\n', res).group(1))

        cmd = f'lddt -c -f -p {stereo_file} -b 15 -a 15 -r 15 -i 0 {fp.name} {ft.name}'
        res = run_cmd(cmd)
        print(res)
        lddt_ca = float(re.search('Global LDDT score:(.+)\n', res).group(1))
    return lddt, lddt_ca

def eval_new(pred, truth, split=None, fasta=None, restr=None, max_mismatch=3, key=None):
    res = {}
    mp, mt, rmsd = align_pred_to_truth(pred, truth, max_mismatch=max_mismatch)
    if (key is None) or ('rmsd' in key):
        res['rmsd'] = rmsd
    if (key is None) or ('tmscore' in key):
        tmscore = cal_tmscore(mp, mt)
        res['tmscore'] = tmscore
    if (key is None) or ('dockq' in key):
        dockq_avg = cal_dockq_avg(mp, mt)
        res['dockq_avg'] = dockq_avg
        if split is not None:
            dockq_dimer = cal_dockq(mp, mt, split)
            res['dockq_dimer'] = dockq_dimer
    if (key is None) or ('lddt' in key):
        lddt, lddt_ca = cal_lddt(mp, mt)
        res['lddt'] = lddt
        res['lddt_ca'] = lddt_ca
    if (key is None) or ('recall' in key):
        if fasta is not None and restr is not None:
            recall, recall_true, restr_num = compute_recall(mp, mt, fasta, restr, return_num=True)
            res['recall'] = recall
            res['recall_true'] = recall_true
            res.update(restr_num)
    return {k: round(v, 3) if v is not None else v for k, v in res.items()}
        


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pred', type=str, help='prediction pdb file')
    parser.add_argument('truth', type=str, help='ground truth pdb file')
    parser.add_argument('--split', type=str, default=None, help='split chains for dockq calculation, base on chain index of truth pdb file, e.g. "0,1:2" means the first and second chains of truth pdb file are used as one part and the third chain is used as the other part for dimer dockq calculation. If not provided, no dimer dockq calculation will be performed')
    parser.add_argument('--fasta', type=str, default=None, help='fasta file for recall calculation')
    parser.add_argument('--restr', type=str, default=None, help='restraint file for recall calculation')
    parser.add_argument('--max_mismatch', type=int, default=3, help='max mismatches allowed for align chains')
    parser.add_argument('--key', type=str, default=None, help='key for evaluation, e.g. "rmsd,tmscore,dockq,lddt,recall"')
    args = parser.parse_args()
    print(args)

    pred = args.pred
    truth = args.truth
    if args.split is not None:
        split = [[int(i) for i in s.split(',')] for s in args.split.split(':')]
    else:
        split = None
    fasta = args.fasta
    restr = args.restr
    
    res = eval_new(pred, truth, split, fasta, restr, max_mismatch=args.max_mismatch, key=args.key)
    print(res)
