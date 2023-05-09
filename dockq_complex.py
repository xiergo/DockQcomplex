import os
import warnings
import argparse

import pandas as pd
import numpy as np
import multiprocessing as mp

from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Model import Model
from dockq.DockQ import calc_DockQ
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', BiopythonWarning)


THREE_TO_ONE ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y', \
    'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A', \
    'GLY':'G', 'PRO':'P', 'CYS':'C', 'UNK': 'X', 'SEC': 'U', 'PYL': 'O'} # SEC 硒代半胱氨酸， PYL 吡咯赖氨酸


def parse_chain(chain):
    seq = ''
    cals = []
    heavls = []
    last_res_idx = None
    for res in chain.get_residues():
        if res.id[0] != ' ':
            continue
        
        res_idx = int(res.id[1])
        if last_res_idx:
            delta_idx = res_idx - last_res_idx 
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
                heavls.append(at.coord)
    ca_pos = np.array(cals)
    heav_pos = np.array(heavls)
    # print(ca_pos.shape)
    assert len(seq) == ca_pos.shape[0], (len(seq), ca_pos.shape)
    mask = np.array([i != 'X' for i in seq], dtype=bool)
    return seq, ca_pos, mask, heav_pos


def cal_rmsd(x1, x2):
    assert x1.shape == x2.shape, (x1.shape, x2.shape)
    assert x1.shape[-1] == 3
    return np.sqrt(((x1 - x2) ** 2).sum(-1).mean())

def get_optimal_transform(src_atoms, tgt_atoms, mask = None):
    assert src_atoms.shape == tgt_atoms.shape, (src_atoms.shape, tgt_atoms.shape)
    assert src_atoms.shape[-1] == 3
    if mask is not None:
        assert mask.dtype == bool
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


def get_mean_pred(pred_pos, mask):
    assert pred_pos.shape[-1] == 3
    if pred_pos.shape[0] != mask.shape[0]:
        return np.array([1e9, 1e9, 1e9]).reshape(1, -1)
    else:
        return pred_pos[mask].mean(0, keepdims=True)


def find_optimal_permutation(x_mean_pred, x_mean_truth):
    '''Find Optimal Permutation'''
    assert x_mean_pred.shape[-1] == 3, x_mean_pred.shape
    assert x_mean_truth.shape[-1] == 3, x_mean_truth.shape
    d_kl = np.sqrt(((x_mean_pred - x_mean_truth[None]) ** 2).sum(-1))
    p_l = []
    for l in range(d_kl.shape[1]):
        d_l = d_kl[:, l]
        best_idx = d_l.argmin()
        p_l.append(best_idx)
        d_kl[best_idx, :] = 1e9
    return p_l

def has_contact(chain1, chain2):
    '''defined as any heavy atom of one chain being within 5A of any heavy atom of the other chain'''
    d = np.expand_dims(chain1, 1) - chain2[None, :]
    dist = np.sqrt((d ** 2).sum(-1))
    return (dist <= 5).any()


def rm_masked_res(chain, mask):
    chain1 = chain.copy()
    for res, m in zip(chain.child_list, mask):
        if not m:
            chain1.detach_child(res.id)
    return chain1


def cal_dockq_pdb(pred_pdb, truth_pdb_dir, pdb_id):
    # print(pdb_id)
    # pred_pdb = f'../output_af2/{pdb_id}/relaxed_model_1_multimer_v3_pred_0.pdb'
    # print(pred_pdb)
    # if not os.path.isfile(pred_pdb):
    #     print(f'{pred_pdb} is not found')
    #     return None
    
    tmp_dir = f'_tmp/{pdb_id}'
    os.makedirs(tmp_dir, exist_ok=True)

    parser = PDBParser(QUIET=True)
    chain_dict = {}
    pred_ca = {}
    pred = parser.get_structure('pred', pred_pdb)[0]
    for chain in pred.get_chains():
        seq, ca_pos, _, _ = parse_chain(chain)
        pred_ca[chain.id] = ca_pos

        if seq in chain_dict:
            chain_dict[seq].append(chain.id)
        else:
            chain_dict[seq] = [chain.id]
        
    ls = []
    for k, v in chain_dict.items():
        ls.append([''.join(v), len(k), k])
    df_pred = pd.DataFrame(ls, columns=['pred_cid', 'seq_len', 'pred_seq'])
    df_pred['num_chains'] = df_pred.pred_cid.map(len)
    # print(df_pred)


    # ground truth pdb
    # merge and rename all chains in gt
    ls = []
    truth_ca = {}
    truth_chain = {}
    PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' # 62

    truth_pdbs = [i.strip() for i in os.popen(f'find {truth_pdb_dir} -name {pdb_id}*pdb').readlines()]
    truth_pdbs = [i for i in truth_pdbs if not os.path.samefile(i, pred_pdb)]
    assert len(truth_pdbs) == len(pred.child_list), f'The number of ground truth chains is not equal to that of prediction: {len(truth_pdbs), len(pred.child_list)}'

    for i, truth_pdb in enumerate(truth_pdbs):
        chain_id = PDB_CHAIN_IDS[i]
        chain = parser.get_structure('truth', truth_pdb)[0].child_list[0]
        truth_chain[chain_id] = chain
        seq, ca_pos, mask, heav_pos = parse_chain(chain)
        truth_ca[chain_id] = [ca_pos, mask, heav_pos]
        
        for i, row in df_pred.iterrows():
            if row.seq_len != len(seq):
                continue
            flag_match = True
            for j, k in zip(row.pred_seq, seq):
                if not (j == k or k == 'X'):
                    flag_match = False
                    break
            if flag_match:
                ls.append([*row, chain_id, truth_pdb, seq, mask])
    df = pd.DataFrame(ls, columns=df_pred.columns.tolist() + ['truth_cid', 'truth_path', 'truth_seq', 'mask']) 
    df['true_seq_len'] = df['mask'].map(np.sum)
    df = df.sort_values(by=['num_chains', 'true_seq_len'], ascending=[True, False])
    print(df)
    df.pop('mask')
    df.to_csv(f'{tmp_dir}/{pdb_id}_info.tsv', sep='\t', index=False)

    truth_cids = df.truth_cid
    anchor_truth = truth_cids[0]
    
    anchors_pred = list(df.pred_cid[0])
    masks = [truth_ca[i][1] for i in truth_cids]
    x_mean_pred = np.concatenate([np.concatenate([get_mean_pred(i, mask) for i in pred_ca.values()])[:, None] for mask in masks], 1)
    # print(x_mean_pred.shape)
    pm_best = []
    rmsd_min = 1e9
    for anchor_pred in anchors_pred:
        ca_p = pred_ca[anchor_pred]
        ca_t, mask, _ = truth_ca[anchor_truth]
        r, t = get_optimal_transform(ca_t, ca_p, mask)
        x_mean_truth = np.concatenate([(truth_ca[i][0][truth_ca[i][1]] @ r + t).mean(0, keepdims=True) for i in truth_cids])
        # print(x_mean_truth.shape)
        pm = find_optimal_permutation(x_mean_pred, x_mean_truth)
        rmsd = cal_rmsd(x_mean_truth, x_mean_pred[pm, range(len(pm))])
        print(anchor_truth, anchor_pred, pm, rmsd)
        if rmsd < rmsd_min:
            rmsd_min = rmsd
            pm_best = pm


    match_table = {}
    for cid_t, cid_p in zip(truth_cids, np.array(list(pred_ca.keys()))[pm_best]):
        cids_p = df.pred_cid[df.truth_cid == cid_t].values[0]
        # print(cid_t, cid_p, cids_p)
        # print(cid_p in cids_p)
        assert cid_p in cids_p, (cid_p, cids_p)
        match_table[cid_t] = cid_p
    with open(f'{tmp_dir}/{pdb_id}_match_table.tsv', 'w') as f:
        f.writelines([f'{df.truth_path[df.truth_cid == k].values[0]}\t{v}\n' for k, v in match_table.items()])


    n_chains = len(match_table)
    dockqls = []
    for i in range(n_chains - 1):
        for j in range(i + 1, n_chains):
            cid_ti = list(match_table.keys())[i]
            cid_tj = list(match_table.keys())[j]
            cont = has_contact(truth_ca[cid_ti][2], truth_ca[cid_tj][2])
            if not cont:
                continue
            cid_pi = match_table[cid_ti]
            cid_pj = match_table[cid_tj]
            file_pred = f'{tmp_dir}/pred_{cid_pi}_{cid_pj}.pdb'
            file_truth = f'{tmp_dir}/truth_{cid_pi}_{cid_pj}.pdb'
            mask_i = truth_ca[cid_ti][1]
            mask_j = truth_ca[cid_tj][1]
            io = PDBIO()
            model_p = Model(0)
            model_p.add(rm_masked_res(pred.child_dict[cid_pi], mask_i))
            model_p.add(rm_masked_res(pred.child_dict[cid_pj], mask_j))
            io.set_structure(model_p)
            io.save(file_pred)
            model_t = Model(0)
            chain_i = truth_chain[cid_ti].copy()
            chain_i.id = cid_pi
            chain_j = truth_chain[cid_tj].copy()
            chain_j.id = cid_pj
            model_t.add(rm_masked_res(chain_i, mask_i))
            model_t.add(rm_masked_res(chain_j, mask_j))
            io.set_structure(model_t)
            io.save(file_truth)
            info = calc_DockQ(file_pred, file_truth)
            info['pdb_id'] = pdb_id
            info['pred_i'] = cid_pi
            info['pred_j'] = cid_pj
            info['truth_i'] = df.truth_path[df.truth_cid == cid_ti].values[0]
            info['truth_j'] = df.truth_path[df.truth_cid == cid_tj].values[0]
            dockqls.append(pd.DataFrame(info, index=[0]))
    
    dockqdf = pd.concat(dockqls, 0)
    cols = ['pdb_id', 'pred_i', 'pred_j', 'DockQ', 'irms', 'Lrms', 'fnat', 'nat_correct', 'nat_total', 'fnonnat', 'nonnat_count', 'model_total', 'chain1', 'chain2', 'len1', 'len2', 'class1', 'class2', 'truth_i', 'truth_j']
    dockqdf = dockqdf[cols]
    print(dockqdf)
    dockqdf.to_csv(f'{tmp_dir}/{pdb_id}_dockq_info.tsv', sep='\t', index=False)
    dockq = dockqdf.DockQ.mean()
    
    return dockq


def main():
    parser = argparse.ArgumentParser(description='Calculate DockQ for protein complex')
    parser.add_argument('pred_pdb', type=str, help='a pdb file containing predicted structures of all chains')
    parser.add_argument('truth_pdb_dir', type=str, help='a directory containing all ground truth pdb files, with each file corresponding to one chain')
    parser.add_argument('pdb_id', type=str, help='PDB id, all files in "truth_pdb_dir" with the pattern "pdb_id***pdb" (excluding pred_pdb) will be recognized as ground truth pdbs')
    args = parser.parse_args()
    dockq = cal_dockq_pdb(args.pred_pdb, args.truth_pdb_dir, args.pdb_id)
    dockq = round(dockq, 5)
    print(f'averaged DockQ: {dockq}')
if __name__ == '__main__':
    main()