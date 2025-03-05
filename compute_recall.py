import re
import os
import tempfile
import numpy as np
import pandas as pd
import pickle
from eval import align_pred_to_truth
from Bio.SeqUtils import seq1
from Bio.PDB.PDBIO import PDBIO


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
    for i in np.where(restraints['interface_mask'])[0]:
        restraints_list.append((i, 8.0))

    # sbr restraints
    for i, j in zip(*np.where(restraints['sbr_mask'])):
        if i>j:
            continue
        distri = restraints['sbr'][i, j]
        cutoff_idx = max(np.where(distri > 1/distri.size)[0])
        print(cutoff_idx)
        cutoff = np.concatenate([BINS, [np.inf]])[cutoff_idx]
        print(f'SBR cutoff: {cutoff}')
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
    else:
        i, j, dist0 = restraint
        dist = dist_mat[i, j] + (1-mask[i]*mask[j])*10000
        satis = dist <= dist0
    return satis, dist

def compute_recall(pred, truth, fasta, restr):
    fasta_dict = reorder_seq_dict(get_fasta_dict(fasta))
    print('fasta_dict:', fasta_dict)

    with open(restr, 'rb') as f:
        restraints = pickle.load(f)

    asym_id = get_asym_id(fasta_dict)
    mp, mt, rmsd = align_pred_to_truth(pred, truth)
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
        satis_gt, dist_gt = check_single_restraint_status(dist_mat_gt, mask_gt, asym_id, restraint)
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
    return recall, recall_true


# if __name__ == '__main__':
#     pred = 