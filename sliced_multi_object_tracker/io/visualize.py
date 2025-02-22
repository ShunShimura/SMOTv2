import os,sys,shutil

from pathlib import Path
from typing import List, Dict, Set
from matplotlib import pyplot as plt
import numpy as np

from sliced_multi_object_tracker.core import *
from sliced_multi_object_tracker.utils import *

def save_detection_map(bboxes:List[Set[Bbox]], ns:int, nf:int, dir:Path) -> None:
    '''
    Save detection map 
        detection map indicates 'where a certain object was captured by bounding-box'
        
    Parameters
    ----------
    bboxes: Time-series of captured bounding boxes at a time. List indicates each time.
    ns: number of slices
    nf: number of frames
    dir: Directory path where detection maps are saved
    '''
    folder = dir / 'detection_map'
    create_or_clear_directory(folder)
    
    # convert bboxes: List[Set[Bbox]] to bboxes: Dict[int, Set[Bbox]] where int is 'id'
    bboxes: Set[Bbox] = {
        b for set in bboxes for b in set if b.id is not None
    }
    bboxes_with_id:Dict[int, Set[Bbox]] = {}
    for b in bboxes:
        if b.id in bboxes_with_id.keys():
            bboxes_with_id[b.id].add(b)
        else:
            bboxes_with_id[b.id] = {b}
    # plot
    for id, tracks in bboxes_with_id.items():
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        detection_frames = set((b.time, b.slice) for b in tracks)
        ax.set_xticks(np.linspace(-1, nf, nf+2), minor=True)
        ax.set_yticks(np.linspace(-1, ns, ns+2), minor=True)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(True, which='both')
        ax.set_xlim(-1, nf+1)
        ax.set_ylim(-1, ns+1)
        ax.set_xlabel("Step: t", fontsize=20)
        ax.set_ylabel("Slice: s", fontsize=20)
        ax.scatter(*zip(*detection_frames), c="red", s=2, zorder=10)
        plt.tight_layout()
        fig.savefig(folder / f'id_{str(id).zfill(3)}.pdf')
        plt.close()

def save_estimation_dists():
    return
