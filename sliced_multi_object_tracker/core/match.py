import os,sys,shutil
from typing import Dict, List, Set, Callable, Tuple, Union
from scipy.optimize import linprog
import numpy as np
import scipy
# import ot

EPS = 1e-6

def xywh2xyxy(bboxes):
    """
    Convert (cx, cy, w, h) to (left, top, right, bottom)

    Parameters
    ----------
    bboxes: (N, 4) array (that means N bounding boxes)

    Returns
    -------
    xyxy_bboxes: (N, 4) array (that means N bounding boxes)
    """
    
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2]
    y2 = bboxes[:, 1] + bboxes[:, 3]
    
    xyxy_bboxes = np.stack([x1, y1, x2, y2], axis=1)
    
    return xyxy_bboxes

def hungarian_matching(instance1: List, instance2: List, f:Callable, threshold:float, **f_kwargs):
    '''
    Matching between instace1 and instance2 
    
    Parameters
    ----------
    instance1: instances that corresponding to the row
    instance2: instances that corresponding to the column
    f (function): determins similarity between instances. (this function is applied for list-like inputs, then returns similarity matrix)
    threshold: the pair whose similarity is lower than this is never matched
    
    Returns
    -------
    match: the list of pair (i, j) where i is index on row and j is index on column. 
    left_row: indexes on row that has no match
    left_col: indexes on column that has no match
    '''
    # exception
    if len(instance1) == 0 or len(instance2) == 0:
        return [], list(range(len(instance1))), list(range(len(instance2)))
    
    n, m = len(instance1), len(instance2)
    
    # make cost_matrix for minimizing
    score_matrix = _score_matrix(instance1, instance2, f, threshold, **f_kwargs)
    cost_matrix = -1.0 * score_matrix
    
    # optimize (minimizing)
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
    matches = np.hstack((row_idx.reshape(-1, 1), col_idx.reshape(-1, 1)))
    
    # split into real-match, left-row, left-col
    match = [match for match in matches if match[0] < n and match[1] < m]
    left_row = [match[0] for match in matches if match[0] < n and match[1] >= m]
    left_col = [match[1] for match in matches if match[0] >= n and match[1] < m]
    return match, left_row, left_col

def many_to_many_matching(
    instance1:List, instance2:List, f:Callable, threshold:float
    ) -> Tuple[Union[List[List[int]], List[int]]]:
    '''
    Conduct many to many matching and return matched indexes.
    '''
    
    # exception
    if len(instance1) == 0 or len(instance2) == 0:
        return [], [], []
    
    n, m = len(instance1), len(instance2)
    
    # make cost_matrix for minimizing
    score_matrix = _score_matrix(instance1, instance2, f)
    cost_matrix = -1.0 * score_matrix
    
    # optimize (minimizing) by LP
        
    # constraints 
    # (n + m constraints) -> A in R^{n+m, n*m}, b in R^{n+m}, each constraint is A_i @ c = b_i
    A_eq = np.zeros((n + m, n * m))
    b_eq = np.zeros((n + m,))
    # Supply constraints
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1 # It means that count up the supply mounts from node-i
        b_eq[i] = min(m / n, 1) # If m < n, each node can supplies multi nodes 
    # Demand constraints
    for j in range(m):
        A_eq[n + j, j::m] = 1 # It means that count up the mounts supplied to node-j
        b_eq[n + j] = min(n / m, 1) # if n < m, each node can obtain from multi nodes
    # Bounds
    bounds = [(0, None)] * (n * m)
    
    # Optimize
    cost = cost_matrix.flatten()
    res = linprog(c=cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Make matched matrix
    if res.success:
        transport_matrix = res.x.reshape(n, m)
        threshold_mask = score_matrix > threshold
        condition = (transport_matrix > EPS) & threshold_mask
        matched_pairs = np.array(np.nonzero(condition)).T.tolist()
        left_row = [i for i in range(n) if not i in [row[0] for row in matched_pairs]]
        left_col = [j for j in range(m) if not j in [row[1] for row in matched_pairs]]
        return matched_pairs, left_row, left_col
    else:
        raise ValueError("Optimization failed.")
    

def _score_matrix(list1: List, list2:List, f:Callable, threshold:float=None,  **f_kwargs):
    '''
    Calculate and make similarity matrix 
    
    Parameters
    ----------
    list1: instances that corresponding to the row
    list2: instances that corresponding to the column
    f (function): determins similarity between instances. (this function is applied for list-like inputs, then returns similarity matrix)
    threshold: If it is not None, the pair whose similarity is lower than this is never matched
    
    Returns
    -------
    np.ndarray: 
        If threshold is not None, (l ,l) array where l = max(n, m) where n is len(list1) and 
        m is len(list2). Addictional elements has the value of threshold.
        Else, (n, m) array that indicates optimal transport.
    '''
    n = len(list1)
    m = len(list2)
    
    # score matrix \in R^{n x m}
    matrix_nm = f(list1, list2, **f_kwargs)
    
    if threshold:
        # add dummy elements; (n, m) -> (n+m, n+m)
        add_dummy_col = np.hstack((matrix_nm, np.full((n, n), threshold)))
        add_dummy_row = np.vstack((add_dummy_col, np.full((m, n+m), threshold)))
        return add_dummy_row
    else:
        return matrix_nm

def intersection_of_union(list1:List[np.ndarray], list2:List[np.ndarray], dim: int = 2, **kwargs):
    '''
    Calculate IoU of bounding boxes
    
    Parameters
    ----------
    list1: list of bounding boxes (np.ndarray) whose format is normalized (cx, cy, w, h)
    list2: list of bounding boxes (np.ndarray) whose format is normalized (cx, cy, w, h)
    dim: dimension on space (dim=3 means (x, y, z, w, h, d))
    
    Returns
    -------
    iou: (n, m) np.array where n is len(list1) and m is len(list2), (i, j) elements indicates IoU between list1[i] and list2[j]
    '''
    bboxes1 = xywh2xyxy(np.array(list1)).reshape((-1, dim * 2))
    bboxes2 = xywh2xyxy(np.array(list2)).reshape((-1, dim * 2))

    coords_b1 = np.split(bboxes1, 2 * dim, axis=1)
    coords_b2 = np.split(bboxes2, 2 * dim, axis=1)

    coords = np.zeros(shape=(2, dim, bboxes1.shape[0], bboxes2.shape[0]))
    val_inter, val_b1, val_b2 = 1.0, 1.0, 1.0
    for d in range(dim):
        coords[0, d] = np.maximum(coords_b1[d], np.transpose(coords_b2[d]))  # top-left
        coords[1, d] = np.minimum(coords_b1[d + dim], np.transpose(coords_b2[d + dim]))  # bottom-right

        val_inter *= np.maximum(coords[1, d] - coords[0, d], 0)
        val_b1 *= coords_b1[d + dim] - coords_b1[d]
        val_b2 *= coords_b2[d + dim] - coords_b2[d]

    iou = val_inter / (np.clip(val_b1 + np.transpose(val_b2) - val_inter, a_min=0, a_max=None) + EPS)
    return iou

def sphere_iou(list1: List[np.ndarray], list2: List[np.ndarray], **kwargs):
    '''
    Calculate IoU of Spheres (x, y, z, r)
    
    Parameters
    ----------
    list1: list of spheres that is np.ndarray of (x, y, z, r)
    list2: list of spheres that is np.ndarray of (x, y, z, r)
    
    Returns
    -------
    ret: (n, m) np.ndarray. (i, j) elements indicates IoU of spheres between list1[i] and list2[j]
    '''
    ret = np.zeros((len(list1), len(list2)))
    for i in range(len(list1)):
        for j in range(len(list2)):
            d = np.sqrt(np.sum((list1[i][:3] - list2[j][:3]) ** 2, axis=0))
            r1, r2 = list1[i][-1], list2[j][-1]
            if r1 + r2 < d:
                v_inter = 0
            elif abs(r1 - r2) < d and d < r1 + r2:
                v_inter = np.pi / (12 * d) * (d - (r1 + r2)) ** 2 * (d ** 2 + 2 * (r1 + r2) * d - 3 * (r1 - r2) ** 2)
            else: # d == 0
                v_inter = 4 * np.pi / 3 * min(r1, r2) ** 3
            ret[i, j] = v_inter / (4 * np.pi / 3 * r1 ** 3 + 4 * np.pi / 3 * r2 ** 3 - v_inter)
    return ret

def count_of_duplicates(bboxes1:List[Set], bboxes2:List[Set], **kwargs):
    '''
    Count number of bounding boxes that exists both lists
    
    Parameters
    ----------
    bboxes1: List of bounding-boxes. Each elements is set of bounding-box
    bboxes2: List of bounding-boxes. Each elements is set of bounding-box
    
    Returns
    -------
    (n, m) np.ndarray: n is len(bboxes1), m is len(bboxes2). (i, j) elements indicates the number of bbox that exists both bboxes1[i] and bboxes2[j]
    
    '''
    return np.array([
        [len(set1 & set2) for set2 in bboxes2] for set1 in bboxes1
    ])
    
def sumIoU(list1:List[Dict[int, np.ndarray]], list2:List[Dict[int, np.ndarray]], **kwargs):
    '''
    Compare how time-series of spheres are similar between elements of list1 and of list2. 
    
    Parameters
    ----------
    list1: List of time-series that consists of spheres (there are possibly lack of time.)
    list2: List of time-series that consists of spheres (there are possibly lack of time.)
    
    Returns
    -------
    ret: (n, m) np.ndarray where n is len(list1) and m is len(list2). Each elemnts are calculated by adding up IoU of spheres
    '''
    def _sum_iou(dict1:Dict[int, np.ndarray], dict2:Dict[int, np.ndarray]):
        score = 0
        for t in dict1.keys():
            if t in dict2.keys():
                score += sphere_iou([dict1[t]], [dict2[t]])[0, 0]
        return score
    
    ret = np.array([
        [_sum_iou(sp1, sp2) for sp2 in list2] for sp1 in list1
    ])
    return ret
