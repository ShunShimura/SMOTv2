''' Evaluate and save the performaces'''

from typing import List, Dict, Set
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

from sliced_multi_object_tracker.io.load import *
from sliced_multi_object_tracker.core import *

EPS = 1e-6

def save_meanIoU(all_predictions:List[Dict[int, np.ndarray]], nf:int, data_loader:DataLoader, dir:Path) -> None:
    '''
    Save prediction performace by text formats.
        IoU means how predicted position overraps ground-truth position.
        'mean' indicates that values are averaged over all tracked spheres.
    
    Parameters
    ----------
    all_predictions: Each shpere's prediction time-series. Key of dict means time.
    nf: number of frames
    data_loader: instance of DataLoader used to extract ground-truth positions
    dir: Directory path where output text file is saved.
    '''
    # GT : [x, y, z, r] (key is time and id)
    ground_truth_positions = data_loader.load_estimation_labels()
    
    # convert predictions to List[int, Dict[np.ndarray]] ( id: {time: array} )
    predictios_with_id: Dict[int, Dict[np.ndarray]] = {}
    for t, predictions in enumerate(all_predictions):
        for id, pred in predictions.items():
            if id in predictios_with_id.keys():
                predictios_with_id[id][t] = pred
            else:
                predictios_with_id[id] = {t: pred}
    predicted_positions: List[Dict[int, np.ndarray]] = list(predictios_with_id.values())
    
    # match preID and gtID to maximize score
    
    n = len(ground_truth_positions)
    m = len(predicted_positions)
    if n == 0 or m == 0:
        with open(dir / 'meanIoU.txt', 'w') as f:
            f.write("gt or pred is None.")
    else:
        match_idx, _, _ = hungarian_matching(
            instance1=ground_truth_positions,
            instance2=predicted_positions,
            f=sumIoU,
            threshold=EPS
        )
        scores = []
        for t in range(nf):
            count = 0
            score_t = 0
            for i, j in match_idx:
                if (not t in ground_truth_positions[i].keys()) or (not t in predicted_positions[j].keys()):
                    continue
                elif (t in ground_truth_positions[i].keys()) and (not t in predicted_positions[j].keys()):
                    count += 1
                else:
                    iou = sphere_iou([ground_truth_positions[i][t]], [predicted_positions[j][t]])[0, 0]
                    score_t += iou
                    count += 1
            scores.append([t, score_t / count])
        with open(dir / 'meanIoU.txt', 'w') as f:
            f.write("\n".join([
                ",".join([str(val) for val in row]) for row in scores
            ]))
            
def save_motmetric(predicted_bboxes:List[Set[Bbox]], nf:int, data_loader:DataLoader, dir:Path):
    '''
    Calculate association performance by AssA, AssPr, AssRe, IDSW and so on.
    
    Parameters
    ----------
    all_bboxes: List[Set[Bbox]]
        The list indicates time-series of re-identified bounding boxes.
    
    nf: int
    
    data_loader: DataLoader
    
    dir: Path
    
    '''
    # Make ground truth as time-series. (frame-level)
    ground_truth: List[Set[Bbox]] = []
    for t in range(nf):
        time_series_of_bboxes = data_loader.load_reidentification_labels_v2(time=t)
        ground_truth.extend(time_series_of_bboxes)
        
    # Calculate association metrics in each threshold \alpha
    asspr, assre, assa, idsw = [], [], [], []
    for alpha in np.arange(0.05, 1.00, 0.05):
        
        # Matching for calculation of score 
        id_pairs:List[List[int]] = []
        for prDets, gtDets in zip(predicted_bboxes, ground_truth):
            prDets, gtDets = list(prDets), list(gtDets)
            matched_idx, left_pr_idx, left_gt_idx = hungarian_matching(
                instance1=[b.bbox for b in prDets],
                instance2=[b.bbox for b in gtDets],
                f=intersection_of_union,
                threshold=alpha
            )
            matched_id = [[prDets[i].id, gtDets[j].id] for i, j in matched_idx]
            left_pr_id = [[prDets[i].id, None] for i in left_pr_idx]
            left_gt_id = [[gtDets[j].id, None] for j in left_gt_idx]
            id_pairs.extend(matched_id + left_pr_id + left_gt_id)
        
        # Calculate AssP, AssR, AssA using id_pairs
        tp_count = 0
        asspr_alpha, assre_alpha = 0, 0
        id_pairs_counter = Counter(map(tuple, id_pairs))
        for pair in id_pairs_counter:
            if None in pair:
                continue
            else:
                tp_count += 1
                tpa = id_pairs_counter.get(pair, 0)
                fpa = id_pairs_counter.get((pair[0], None), 0)
                fna = id_pairs_counter.get((None, pair[1]), 0)
                asspr_alpha += tpa / (tpa + fpa) if tpa + fpa > 0 else 0
                assre_alpha += tpa / (tpa + fna) if tpa + fna > 0 else 0
        asspr_alpha /= tp_count
        assre_alpha /= tp_count
        assa_alpha = asspr_alpha * assre_alpha /\
            (asspr_alpha + assre_alpha - asspr_alpha * assre_alpha)

        # Calculate IDSW
        idsw_alpha = 0
        unique_pairs = set(map(tuple, id_pairs))
        prids = [row[0] for row in unique_pairs]
        for prid in prids:
            idsw_alpha += len([ret for row in unique_pairs if (ret := row[0]) == prid])

        # Save
        asspr.append(asspr_alpha)
        assre.append(assre_alpha)
        assa.append(assa_alpha)
        idsw.append(idsw_alpha)

    # Row indicates alpha {0.05, 0.10, ..., 0.95} and average over alpha
    alphas =['alpha:0.' + str(val).zfill(2) for val in range(5, 100, 5)] + ['Avg.'] 
    data = {
        'AssPr': asspr + [sum(asspr) / len(asspr)],
        'AssRe': assre + [sum(assre) / len(assre)],
        'AssA': assa + [sum(assa) / len(assa)],
        'IDSW': idsw + [sum(idsw) / len(idsw)]
    }
    df = pd.DataFrame(data, index=alphas)
    df.to_csv(dir / 'motmetrics.txt', header=True, index=True)
    
def cm_association():
    '''
    Return True Positive Association (TPA), False Positive Association (FPA), False Negative 
    Association (FNA).
    '''
    
def score_maximizing_hota(prDets:List[Bbox], gtDets:List[Bbox], preliminary_pairs:List[List[int]], alpha:float) -> np.ndarray:
    '''
    Return matching score matrix according to eq.15 on HOTA paper (see 
    https://link.springer.com/article/10.1007/s11263-020-01375-2). 
    '''
    n = len(prDets)
    m = len(gtDets)
    
    # Add first term 1 / \epsilon
    score:np.ndarray = np.full((n, m), 1 / EPS)
    
    # Add second term A_max using preliminary matching pairs
    for i in range(n):
        for j in range(m):
            prid, gtid = prDets[i].id, gtDets[j].id
            tpa = preliminary_pairs.count([prid, gtid])
            union = preliminary_pairs.count([prid, None]) +\
                    preliminary_pairs.count([None, gtid]) - tpa
            score[i, j] += tpa / union if union != 0 else 0
            
    # Add third term \epsilon S_{ij}. If S < alpha, score = 0
    score_iou = intersection_of_union(
        list1=[b.bbox for b in prDets],
        list2=[b.bbox for b in gtDets],
        dim=2
    )
    alpha_mask = score_iou > alpha
    result = np.where(alpha_mask, score + score_iou, 0)
    
    return result
