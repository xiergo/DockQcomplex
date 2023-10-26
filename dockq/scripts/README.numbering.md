

# fix_numbering performs a sequence alignment between the model and the native based on the "Needleman–Wunsch algorithm" and slides the unaligned residues appropreately in both structure.

usage: ./fix_numbering.pl <model.pdb> <template.pdb> <read_seq_from_atom_in_residue (if def)>
 OUTFILE: model.fixed

e.g., ./fix_numbering.pl model_unfixed.pdb native_unfixed.pdb

>MODEL:
SVIHPLQNLLTSRDGSLVFAIIKNCILSFKYQSPNHWEFAGKWSDDFPIYSYIRNLRLTSDESRLIACADSDKSLLVFDVDKTSKNVLKLRKRFCFSKRPNAISIAEDDTTVIIADKFGDVYSIDINSIPEEKFTQEPILGHVSMLTDVHLIKDSDGHQFIITSDRDEHIKISHYPQCFIVDKWLFGHKHFVSSICCGKDYLLLSAGGDDKIFAWDWKTGKNLSTFDYNSLIKPYLNDQHLA-PIIEFAVSKIIKSKNLPFVAFFVEATKCIIILEMSEKQKGDLALKQIITFPYNVISLSAHNDEFQVTLDNKESSGVQKNFAKFIEYNLNENSFVVNNEKSNEFDSAIIQSVQGDSNLVTKKEEIYPLYNVSSL-------PQDMDWSKLYPYYK-----QMTKKVTIADIGCGFGGLMIDLSPAFPEDLILGMEIRVQVTNYVEDRIIALRNNTASKHGFQNINVLRGNAMKFLPNFFEKGQLSKMFFCFPDPHKARIITNTLLSEYAYVLKEGGVVYTITDVKDLHEWMVKHLEEHPLFERLSKEWEENDECVKIMRNATEEGKKVERKKGDKFVACFTRLPTPAIL

>NATIVE:
SVIHPLQNLLTSRDGSLVFAIIKNCILSFKYQSPNHWEFAGKWSDDFPIYSYIRNLRLTSDESRLIACADSDKSLLVFDVDKTSKNVLKLRKRFCFSKRPNAISIAEDDTTVIIADKFGDVYSIDINSIPEEKFTQEPILGHVSMLTDVHLIKDSDGHQFIITSDRDEHIKISHYPQCFIVDKWLFGHKHFVSSICCGKDYLLLSAGGDDKIFAWDWKTGKNLSTFDYNSLIKPYLNDQHLAPPIIEFAVSKIIKSKNLPFVAFFVEATKCIIILEMSEKQKGDLALKQIITFPYNVISLSAHNDEFQVTLDNKESSGVQKNFAKFIEYNLNENSFVVNNEKSNEFDSAIIQSVQGDSNLVTKKEEIYPLYNVSSLQLEYPVSPQDMDWSKLYPYYKNAENGQMTKKVTIADIGCGFGGLMIDLSPAFPEDLILGMEIRVQVTNYVEDRIIALRNNTASKHGFQNINVLRGNAMKFLPNFFEKGQLSKMFFCFPDP---RIITNTLLSEYAYVLKEGGVVYTITDVKDLHEWMVKHLEEHPLFERLSKEWEENDECVKIMRNAT-----------DKFVACFTRLPTPAIL

YOU NEED TO HAVE 'needle' INSTALLED AS PART OF THE EMBOS PACKAGE: http://emboss.sourceforge.net/apps/release/6.6/emboss/apps/needle.html

# renumber_pdb.pl renumbers residues from 1 for each of the two chains in a complex PDB file containing multiple chains for either the receptor or the ligand or both. 

usage: ./renumbering.pl <PDB>
 OUTFILE: PDB.renum

