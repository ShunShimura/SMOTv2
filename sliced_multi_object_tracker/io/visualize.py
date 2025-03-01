import os,sys,shutil

from pathlib import Path
from typing import List, Dict, Set
from matplotlib import pyplot as plt
import numpy as np

from sliced_multi_object_tracker.core import *
from sliced_multi_object_tracker.utils import *
from sliced_multi_object_tracker.io.load import DataLoader

EPS = 1e-6

def save_detection_map(bboxes:List[Set[Bbox]], ns:int, nf:int, dir:Path) -> None:
    '''
    Save detection map 
        detection map indicates 'where a certain object was captured by bounding-box'
        
    Parameters
    ----------
    bboxes: Time-series (frame-level) of captured bounding boxes at a time. List indicates each time.
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

def save_posterior(
    all_predictions:List[Dict[int, np.ndarray]],
    all_filterings:List[Dict[int, List[np.ndarray]]],
    number_of_frame:int,
    data_loader:DataLoader,
    dir:Path
    ) -> None:
    ''' Visualize the posterior and groundtruth '''
    # Ground truth of states whose structure is {id: {time: state}}
    gt_states_dict:Dict[int, Dict[int, np.ndarray]] = data_loader.load_estimation_labels()
    gt_states = list(gt_states_dict.values())

    # Convert prediction's structure 
    pr_pos_dict:Dict[int, Dict[int, np.ndarray]] = {}
    for t, predictions in enumerate(all_predictions):
        for id, pred in predictions.items():
            if id in pr_pos_dict.keys():
                pr_pos_dict[id][t] = pred
            else:
                pr_pos_dict[id] = {t: pred}
    pr_pos:List[Dict[int, np.ndarray]] = list(pr_pos_dict.values())
    pr_pos_key:List[int] = list(pr_pos_dict.keys())
    
    # Convert filterings to {id: {time: [means, vars]}}
    filterings_dict:Dict[int, Dict[int, List[np.ndarray]]] = {}
    for t, filterings in enumerate(all_filterings):
        for id, post in filterings.items():
            if id in filterings_dict.keys():
                filterings_dict[id][t] = post
            else:
                filterings_dict[id] = {t: post}
    
    # Make GT of positions
    gt_pos = [
        {t: value[::2] for t,value in gt.items()}
        for gt in gt_states
    ]
    
    # ID Matching between predictions and groundtruths
    matched_idx, _, _ = hungarian_matching(
        instance1=pr_pos,
        instance2=gt_pos,
        f=sumIoU,
        threshold=EPS
    )
    for i, j in matched_idx:
        id:int = pr_pos_key[i]
        posteriors:Dict[int, List[np.ndarray]] = filterings_dict[id]
        means = {t: v[0] for t,v in posteriors.items()}
        vars = {t: v[1] for t, v in posteriors.items()}
        gts:Dict[int, List[np.ndarray]] = gt_states[j]
        plot_mean_and_variance(means, vars, gts, nf=number_of_frame, output=dir / f'id{str(id).zfill(3)}.pdf')

def plot_mean_and_variance(means:Dict[int, np.ndarray], vars:Dict[int, np.ndarray], label_means:Dict[int, np.ndarray], output:Path, nf:int):
    fig = plt.figure(figsize=(10, 6))
    fig_labels = ["x", 'y', 'z', 'r']
    for i in range(3):
        ax0 = fig.add_subplot(2, 4, i+1)
        ax1 = fig.add_subplot(2, 4, i+5)
        time = [t for t in means.keys() if t in label_means.keys()]
        m0 = np.array([means[t][2*i] for t in time])
        s0 = np.array([vars[t][2*i]**0.5 for t in time])
        g0 = np.array([label_means[t][2*i] for t in time])
        m1 = np.array([means[t][2*i+1] for t in time])
        s1 = np.array([vars[t][2*i+1]**0.5 for t in time])
        g1 = np.array([label_means[t][2*i+1] for t in time]) * 100
        
        ax0.plot(time, m0, label=f"mean({fig_labels[i]})", color="blue")
        ax0.plot(time, g0[:len(m0)], label=f"ground-truth({fig_labels[i]})", color="black")
        ax0.fill_between(time, m0-2*s0, m0+2*s0, color="blue", alpha=0.2)
        ax0.legend(loc="lower right")
        ax0.set_xlim(0, nf)
        ax0.set_ylim(np.min(m0)-3*np.mean(s0), np.max(m0)+3*np.mean(s0))
        ax1.plot(time, m1, label=f"mean({fig_labels[i]}')", color="blue")
        ax1.plot(time, g1, label=f"ground-truth({fig_labels[i]}')", color="black")
        ax1.fill_between(time, m1-2*s1, m1+2*s1, color="blue", alpha=0.2)
        ax1.legend(loc="lower right")
        ax1.set_xlim(0, nf)
        ax1.set_ylim(np.min(m1)-3*np.mean(s1), np.max(m1)+3*np.mean(s1))
    # plot of (r,)
    ax = fig.add_subplot(2, 4, 4)
    m = np.array([means[t][-1] for t in time])
    s = np.array([vars[t][-1] for t in time])
    g = np.array([label_means[t][-1] for t in time])
    
    ax.plot(time, m, label=f"mean({fig_labels[-1]})", color="blue")
    ax.plot(time, g, label=f"ground-truth({fig_labels[-1]})", color="black")
    ax.fill_between(time, m-2*s, m+2*s, color="blue", alpha=0.2)
    ax.legend(loc="lower right")
    ax.set_xlim(0, nf)
    ax.set_ylim(np.min(m)-3*np.mean(s), np.max(m)+3*np.mean(s))
    fig.savefig(output)
    plt.close()