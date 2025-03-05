from pymol import cmd
import re
import os
import tempfile
import numpy as np
import pandas as pd
from eval import align_pred_to_truth
from Bio.SeqUtils import seq1
from Bio.PDB.PDBIO import PDBIO
import utils
BINS = np.arange(4, 33, 1)


class PymolAlign:
    def __init__(self, pred_dict, truth, fasta, restraints, anchor_idx, chain_colors, outdir, xl_type_colors=None):
        # pred_dict is a dictionary that maps method to the predicted structure file path
        # truth is the ground truth structure file path
        # fasta is the fasta file path
        # chain_colors is a list of colors for each chain in the fasta file, or a single color for all chains
        # anchor_idx is the index of the chain that should be used as the anchor for alignment
        # restraints is a dictionary that maps restraint type to a boolean mask of restraints
        # outdir is the output directory for saving the results
        # xl_type_colors is a dictionary that maps xl cutoff length to a color for the restraints, only show in GT model
        assert 'GT' not in pred_dict, 'GT should not be in the pred_dict'
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        model_dict = {'GT': None}
        for method, pred in pred_dict.items():
            pred = utils.split_chains(fasta, pred)
            print(f'{method} pred goushi:', pred)
            mp, mt, rmsd = align_pred_to_truth(pred, truth)
            print(f'{method}: RMSD: {rmsd}')
            model_dict[method] = mp
        model_dict['GT'] = mt
        self.model_dict = model_dict
        self.xl_type_colors = xl_type_colors
        fasta_dict = utils.get_fasta_dict(fasta)
        print('fasta_dict before reorder:', fasta_dict)
        self.fasta_dict = utils.reorder_seq_dict(fasta_dict)
        print('fasta_dict after reorder:', self.fasta_dict)
        self.asym_id = utils.get_asym_id(self.fasta_dict)
        self.residues_dict = {k: utils.get_residues_from_fasta_dict(v, self.fasta_dict) 
                              for k, v in model_dict.items()}
        self.chains_dict = self._get_chains()
        print('chains_dict:', self.chains_dict)
        self.anchor_idx = anchor_idx
        print('anchor_idx:', self.anchor_idx)
        self.chain_colors = chain_colors
        print('chain_colors:', self.chain_colors)
        self.restraints_list = utils.parse_restraints(restraints)
        cmd.delete('all')

    def _get_chains(self):
        # this function should return the list of chain ids in the model for each method
        # the order of chains should be the same as the order in the fasta file
        chains_dict = {}
        for method, residues in self.residues_dict.items():
            cids = [res.parent.id for res in residues if res is not None]
            cids_uniq = list(dict.fromkeys(cids))
            chains_dict[method] = cids_uniq
        return chains_dict

    def color_chains(self):
        # this function should color the chains in pymol
        print('Coloring chains')
        if isinstance(self.chain_colors, str):
            colors = [self.chain_colors] * len(self.chains_dict['GT'])
        else:
            colors = self.chain_colors
        for method in self.model_dict:
            for cid, color in zip(self.chains_dict[method], colors):
                # if method == 'GT':
                #     color = 'gray'
                cmd.color(color, f'{method} and chain {cid}')
                print(f'{method} chain {cid} colored {color}')
        print('Chains colored')

    def check_restraints_status(self):
        # this function should check the status of restraints for all methods
        # GT model should be the fisrt model in the list, and should be used as the reference model
        # satisfied restraints should be marked as True, 
        # unsatisfied restraints should be marked as False.
        # the function should return a dictionary that maps method to a dictionary of restraints:restraints status
        print('Checking restraints status')
        restraints_satis = {}
        summary_ls = [] # for saving the dataframe
        restraints_status_ls = [] # for saving the dataframe
        for method, residues in self.residues_dict.items():
            pseudo_beta_coords, mask = utils.get_pseudo_beta(residues)
            dist_mat = np.sqrt(np.sum((pseudo_beta_coords[None] - pseudo_beta_coords[:, None])**2, axis=-1))
            restraints_satis[method] = {}
            satis_num = 0
            tot_num = 0
            satis_correct_num = 0
            correct_num = 0
            for restraint in self.restraints_list:
                satis, dist = utils.check_single_restraint_status(dist_mat, mask, self.asym_id, restraint)
                restraints_satis[method][restraint] = satis
                correct = satis if method == 'GT' else restraints_satis['GT'][restraint]

                # for saving the dataframe
                restraints_status_ls.append({
                    'Method': method,
                    'Restraint': restraint,
                    'Type': 'SBR' if len(restraint) == 3 else 'Interface',
                    'Satisfied': satis,
                    'Correct': correct,
                    'Distance': dist,
                })

                # for computing the recall
                if satis:
                    satis_num += 1
                tot_num += 1

                if correct:
                    correct_num += 1
                    if satis:
                        satis_correct_num += 1
            recall = satis_num / tot_num if tot_num > 0 else 0
            recall_true = satis_correct_num / correct_num if correct_num > 0 else 0

            # for saving the dataframe
            summary_ls.append({
                'Method': method,
                'total_restraints': tot_num,
                'total_restraints_satis': satis_num,
                'correct_restraints': correct_num,
                'correct_restraints_satis': satis_correct_num,
                'Recall': recall,
                'Recall_true': recall_true,
            })
            print(f'{method}: Recall: {recall}, Recall_true: {recall_true}')
        df_restraints_summary = pd.DataFrame(summary_ls)
        df_restraints_summary.to_csv(f'{self.outdir}/restraints_summary.tsv', sep='\t', index=False)
        df_restraints_status = pd.DataFrame(restraints_status_ls)
        df_restraints_status.to_csv(f'{self.outdir}/restraints_status.tsv', sep='\t', index=False)
        self.restraints_satis = restraints_satis
        print('Restraints check done')
        
    def load_models(self):
        # this function should load the models into pymol
        # color the GT model in gray
        # color the other models in specific colors
        print('Loading models')
        for method, model in self.model_dict.items():
            with tempfile.NamedTemporaryFile('w', suffix='.pdb') as f:
                io = PDBIO()
                io.set_structure(model)
                io.save(f.name)
                cmd.load(f.name, f'{method}')
                print(f'{method} loaded')
        print('Models loaded')
    
    def align_models(self):
        # this function should align the anchor chain of the models to that of the GT model
        for method in self.model_dict.keys():
            if method == 'GT':
                cmd.center(f'{method}')
                continue
            cmd.align(f'{method} and chain {self.chains_dict[method][self.anchor_idx]}', f'GT and chain {self.chains_dict["GT"][self.anchor_idx]}')
            print(f'{method} chain {self.chains_dict[method][self.anchor_idx]} aligned to GT chain {self.chains_dict["GT"][self.anchor_idx]}')

    def _choose(self, residue, level):
        # this function should generate the string for choosing the object of residue at the given level in pymol
        # level can be 'chain', 'residue', 'pseudo-beta'
        if residue is None:
            return ''
        string = f'chain {residue.parent.id}'
        if level == 'chain':
            return string
        string += f' and resi {residue.id[1]}'
        if level =='residue':
            return string
        if 'CB' in residue:
            string += f' and name CB'
        elif 'CA' in residue:
            string += f' and name CA'
        else:
            string += f' and name C'
        if level == 'pseudo-beta':
            return string
        raise ValueError(f'Invalid level: {level}')

    def show_restraints(self):
        # this function should show the restraints in pymol
        # Satisfied restraints should be shown in blue, 
        # unsatisfied restraints should be shown in red.
        # incorrect restraints should be shown in dashed lines for sbr restraints .
        # correct restraints should be shown in solid lines for sbr restraints.
        # correct restraints should be shown in sticks for interface  residues.
        print('Showing restraints')
        # before showing restraints, make a copy of GT model
        cmd.create('GT_copy', 'GT')
        cmd.color('gray', 'GT_copy')
        for method, residues in self.residues_dict.items():
            for restraint in self.restraints_list:
                statis = self.restraints_satis[method][restraint]
                statis_color = 'blue' if statis else'red'
                correct = self.restraints_satis['GT'][restraint]


                if len(restraint) == 2:
                    i, dist = restraint
                    if residues[i] is None:
                        continue
                    cmd.color(statis_color, f'{method} and {self._choose(residues[i], "residue")}')
                    if correct:
                        cmd.show('sticks', f'{method} and {self._choose(residues[i], "residue")}')
                else:
                    i, j, dist = restraint
                    if residues[i] is None or residues[j] is None:
                        continue
                    dist_name = f'{method}_d{i}_{j}'
                    cmd.distance(dist_name, f'{method} and {self._choose(residues[i], "pseudo-beta")}', 
                                 f'{method} and {self._choose(residues[j], "pseudo-beta")}') # label=0 to hide the distance
                    
                    if method == 'GT':
                        if self.xl_type_colors is None:
                            xl_color = 'blue'
                        else:
                            xl_color = self.xl_type_colors.get(dist, 'blue')
                        cmd.color(xl_color, dist_name)
                    else:
                        cmd.color(statis_color, dist_name)
                    
                    cmd.set('label_color', 'black', dist_name)
                    cmd.set('dash_width', 5, dist_name) # default dash_width is 2, change it to 5 to make the line thicker
                    # cmd.set('dash_transparency', 0.4, dist_name)
                    if not correct:
                        # dashed line for incorrect restraints
                        cmd.set('dash_gap', 3, dist_name)
                        cmd.set('dash_length', 1, dist_name)
                    else:
                        # solid line for correct restraints
                        cmd.set('dash_gap', 0, dist_name)
                        cmd.set('dash_length', 1, dist_name)
                        
            print(f'{method} restraints shown')
        print('Restraints shown')

    def save_pse(self):
        # this function should save the pse file for the aligned models
        self.load_models()
        self.align_models()
        self.color_chains()
        self.check_restraints_status()
        self.show_restraints()
        cmd.disable('*')
        cmd.enable('GT*')
        cmd.disable('GT_copy')
        print('Saving pse file')
        cmd.save(f'{self.outdir}/aligned.pse')
        print(f'Pse file saved to {self.outdir}/aligned.pse')


    