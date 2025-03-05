from Bio.Align import PairwiseAligner
import numpy as np
def match_strings(str1, str2, max_errors=3):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.wildcard = 'X'
    aligner.match_score = 0
    aligner.mismatch_score = -1
    aligner.target_gap_score = -100

    aligner.query_internal_gap_score = -100
    aligner.query_end_gap_score = 0

    alignments = aligner.align(str1, str2)
    best_alignment = alignments[0]

    is_match = best_alignment.score >= (-max_errors)
    print(best_alignment.score)
    
    if is_match:
        if best_alignment.score<0:
            print(f'Warning: There are {-int(best_alignment.score)} mismatch between target and query')
            print(best_alignment)
        x = np.where([i!='-' for i in best_alignment[1]])[0]
        span = (x.min(), x.max()+1)
        mask = np.array([i==j for i, j in zip(*best_alignment)], dtype=bool)
    else:
        span=None
        mask=None

    return is_match, span, mask

# Example usage:
str1 = "ABCDERTYG"
str2 = "BCXEY"
is_match, span, mask = match_strings(str1, str2)
print(is_match)
print(span)
print(mask)
