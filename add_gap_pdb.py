import os
import sys

def generate_pdb(pdb_file, seq_len):
    # pdb_id = '8DZE_F'
    # pdb_file = f'pdb_renum/{pdb_id}_renum.pdb'
    outfile = pdb_file.replace('.pdb', '_no_gap.pdb')
    print(f'{pdb_file}: {seq_len}')
    new_lines = []
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    dic = {}
    template = None
    for line in lines:
        if not line.startswith('ATOM'):
            continue
        if not template:
            template = line
        index = int(line[22:26])
        if index not in dic:
            dic[index] = []
        dic[index].append(line[11:])
    for i in range(seq_len):
        if i+1 in dic:
            new_lines.extend(dic[i+1])
        else:
            new_lines.append(f'  N   UNK{template[20:22]}{i+1:>4}       0.000   0.000   0.000{template[54:73]}    N\n')
            new_lines.append(f'  C   UNK{template[20:22]}{i+1:>4}       0.000   0.000   0.000{template[54:73]}    C\n')
    with open(outfile, 'w') as f:
        f.writelines([f'ATOM  {i+1:>5}{line}' for i, line in enumerate(new_lines)])
    print(f'No gap pdb: {outfile}')


if __name__ == '__main__':
    pdb_file = sys.argv[1]
    seq_len = sys.argv[2]
    generate_pdb(pdb_file, seq_len)