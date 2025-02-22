import os,sys,shutil
from typing import List, Dict, Set
from pathlib import Path
import numpy as np
import logging

from sliced_multi_object_tracker.core import *
from sliced_multi_object_tracker.filter import *
from sliced_multi_object_tracker.tracker.tracker import *
from sliced_multi_object_tracker.tracker.bbox_tracker import *

EPS = 1e-6
logger = logging.getLogger('mylogger')
    
class BaseSlicedMultiObjectTracker(BaseMultiObjectTracker):
    '''
    Base class corresponding sliced bounding-boxes input
    
    Attributes
    ----------
    ns: number of slice
    nf: number of frame (= duration)
    model: state-space model corresponding multiple bounding-box input and sphere-like state
    
    Methods
    -------
    step: Main process of tracking
    _save_reid: save bounding-boxes (in the timing this function used, bboxes have obtained ID)
    '''
    def __init__(
        self,
        ns: int,
        nf: int,
        model: SlicedModel,
    ):
        super().__init__() # make self.single_trackers, id_counter
        self.ns = ns
        self.nf = nf
        self.model = model
        
        self.all_bboxes: List[Set[Bbox]] = [] # for save ReID results
        self.all_predictions: List[Dict[int, np.ndarray]] = [] # for save Estimation results
        
    def step(self, sliced_obs:Dict[int, List[Bbox]]):
        return NotImplementedError
            
    def _save_reid(self, sliced_obs:Dict[int, List[Bbox]]) -> None:
        # for ReID
        self.all_bboxes.append(
            {bbox for bbox_list in sliced_obs.values() for bbox in bbox_list}
        )
    
class SlicedMultiObjectTracker(BaseSlicedMultiObjectTracker):
    '''
    Proposed method class.
    This tracker conducts ReID and Estimation. So this can use no more than two kalman-filters
    (one is used for ReID, and another is used for estimation)
    Estimator used SliceKalmanFilter (is also proposed method) to estimate sphere-like state-vector.
    
    Attributes
    ----------
    ns: number of slice
    nf: number of frame (length of discrete time)
    temporal_max_staleness: threshold to terminate tracking 
    spacio_max_staleness: threshold to terminate depth-directinal track on Depth-SORT
    model: constract state-space-model on SliceKalmanFilter
    bbox_model: constract state-space-model whose state is 2D-like (x, y, w, h) and their differentiations
    mode: determine proposed method's mode (whether it uses bbox_model for time-directional matching or not, and whether uses Depth-SORT or not)
    
    Methods
    -------
    step: Main process
        1. Time-directional matching (slice-wise matching): In each slice, compare predictions and observations and ReID to observations
        2. Depth-directional matching and integration with 1.: Split bounding-boxes to boxes derived from identical one and give ID by reffering 1.
        3. Other processes (update, generate, stale, predict)
            In prediction, use SliceKalmanFilter and output predictions (\mu_{t+1|t})
    gt_step: Use ground-truth bounding-boxes that has ID that derives it 
    active_trackers: Return single trackers whose staleness is less than max_staleness
    '''
    def __init__(
        self,
        ns: int,
        nf: int,
        temporal_max_staleness: int,
        spacio_max_staleness: int,
        model: SlicedModel,
        bbox_model: BboxModel,
        mode: str = 'SD'
    ):
        super().__init__(ns, nf, model)
        self.mode = mode
        self.bbox_model = bbox_model
        self.max_staleness = {'time':temporal_max_staleness, 'space':spacio_max_staleness}
        
    def step(self, sliced_obs:Dict[int, List[Bbox]]):
        '''
        Main process of tracking
        
        mode SD: Use 2D-like model for time-directional tracking and Depth-SORT for ReID
        mode KD: Use 3D-like model for time-directional tracking and Depth-SORT for ReID
        mode K: Use 3D-like model for time-directional tracking and use it to eventual ReID results as it is.
        
        1. Slice-wise matching
            To find bounding-boxes that is might derived from objects tracked at time-t using predictions
                'S' uses slice-wise SingleBboxTracker to predict (they have the same ID)
                'K' uses SliceKalmanFilter to predict
                
        2. Depth-directional matching and integrating
            To split bounding-boxes observed at time-t into boxes derived from one object
            This process is only conducted mode concludes 'D'
            Then, assign IDs tracked at time-t to these by reffering 1. results (see intersection of boxes on each set)
            
        3. Update, stale, generate
            Update: update each KalmanFilters (2D and 3D) using assined bounding-boxes
            Stale: increse staleness of single-trackers that is in tracking and do not obtain new observations
            Generate: Use left bounding boxes obtained by Depth-SORT to generate new single-trackers
            
        4. Predict
            Make predictions of (x, y, z, r) by using 3D-like KalmanFilter (this process also proceed kalman procedures)
            
        Parameters
        ----------
        sliced_obs: Bounding-boxes obtained by detection as the dictionary whose key is slice-index
        '''
        # 1. slice-wise matching
        
        active_trackers: List[SlicedSingleObjectTracker] = self.active_trackers()
        identified_bboxes: List[Set[Bbox]] = [set() for _ in active_trackers]
        for s in range(self.ns):
            # Because index number variates depending on slice, preserve the index on active_trackers 
            predicted_idx_and_bboxes = [[i, ret] for i, at in enumerate(active_trackers) if (ret := at.predicted_bboxes[s]) is not None]
            predicted_idx = [row[0] for row in predicted_idx_and_bboxes]
            predicted_bboxes: List[np.ndarray] = [row[1] for row in predicted_idx_and_bboxes]
            observed_bboxes: List[np.ndarray] = [b.bbox for b in sliced_obs[s]]
            
            # matching 
            matched_idx, _, _, = hungarian_matching(
                predicted_bboxes,
                observed_bboxes,
                f=intersection_of_union,
                threshold=EPS
            )
            # make temporary identified bounding boxes list
            for i, j in matched_idx:
                identified_bboxes[predicted_idx[i]].add(
                    sliced_obs[s][j]
                )
                
        # 2. detph-sort and intergration
        if self.mode in ['SD', 'KD']: # mode 'K' does not use Depth-SORT
            dsort_identified_bboxes: List[Set[Bbox]] = depth_sort(
                sliced_obs,
                bbox_model=self.bbox_model,
                ns=self.ns,
                max_staleness=self.max_staleness['space']
            )
            logger.info(f"Found {len(dsort_identified_bboxes)} objects by Depth-SORT")
            matched_idx, _, new_idx = hungarian_matching(
                identified_bboxes,
                dsort_identified_bboxes,
                f=count_of_duplicates,
                threshold=EPS
            )
            new_identified_bboxes: List[Set[Bbox]] = [set() for _ in active_trackers]
            for i, j in matched_idx:
                new_identified_bboxes[i] = dsort_identified_bboxes[j]
            identified_bboxes = new_identified_bboxes
            
        # 3. update, stale, generate and save
        
        # update
        updated_idx_list: List[int] = [] # for stale 
        for i, bboxes in enumerate(identified_bboxes):
            if len(bboxes) > 0:
                active_trackers[i].update(bboxes)
                updated_idx_list.append(i)
                
        # generate
        if self.mode == 'K':
            # find bboxes that consists of ones which is not used for update
            new_idx = unused_dsort_idx(dsort_identified_bboxes, identified_bboxes)
        for idx in new_idx:
            # if you use SORT, multi_state is True
            multi_state = True if self.mode in ['SD', 'S'] else False
            self.single_trackers.append(
                SlicedSingleObjectTracker(
                    id=self.id_counter.get_id(),
                    model=self.model,
                    ns=self.ns,
                    bbox_model=self.bbox_model,
                    obs0=dsort_identified_bboxes[idx],
                    multi_state=multi_state,
                )
            )
        
        # stale
        for i, at in enumerate(active_trackers):
            if i in updated_idx_list:
                continue
            else:
                at.stale()
                
        # predict and save
        self.all_predictions.append(
            {at.id : at.predict() for at in self.active_trackers()}
        )
        self._save_reid(sliced_obs=sliced_obs)
        
    def gt_step(self, gt_bboxes:Dict[int, Set[Bbox]]):
        '''
        If only evaluate estimation's performance, use ground-truth ReID results.
            Each bounding-boxes already have deriving ID
            
        Parameters
        ----------
        gt_bboxes: dictionary of bounding-boxes with deriving ID sorted by slice-index 
        '''
        # make id_to_idx map of active_trackers 
        active_trackers = self.active_trackers()
        map_to_idx = {st.id: i for i, st in enumerate(active_trackers)}
        
        # use ground-truth to update
        stale_idx = set(range(len(active_trackers)))
        for id in gt_bboxes.keys():
            if id in map_to_idx.keys():
                active_trackers[map_to_idx[id]].update(obs=gt_bboxes[id])
                stale_idx.remove(map_to_idx[id])
            else:
                # generate
                self.single_trackers.append(
                    SlicedSingleObjectTracker(
                        id=self.id_counter.get_id(),
                        model=self.model,
                        bbox_model=self.bbox_model,
                        ns=self.ns,
                        obs0=gt_bboxes[id],
                        multi_state=False
                    )
                )
        # stale
        for i in stale_idx:
            active_trackers[i].stale()
            
        # predict and save
        self.all_predictions.append(
            {at.id : at.predict() for at in self.active_trackers()}
        )
        sliced_obs: Dict[int, List[Bbox]] = {s: [] for s in range(self.ns)}
        for bboxes_set in gt_bboxes.values():
            for bbox in bboxes_set:
                sliced_obs[bbox.slice].append(bbox)
        self._save(sliced_obs=sliced_obs)
        
    def active_trackers(self) -> List[BaseSingleObjectTracker]:
        '''
        Return active trackers. 'Active' means staleness is less than threshold (that is not missing)
        
        Returns
        -------
        List of single-trackers that is active
        '''
        return [sot for sot in self.single_trackers if not sot.is_stale(self.max_staleness['time'])]

        
class SlicedSingleObjectTracker(BaseSingleObjectTracker):
    '''
    Tracker of single object that corresponds to multiple bounding-box input and can estimate and predict (x, y, z, r) 
    
    Attributes
    ----------
    id: Number to identify
    model: 3D-like state-space-model
    ns: number of slice
    bbox_model: The state-space-model used for prediction on time-directional matching (mode 'K' and 'KD' don't use this)
    multi_state: Wether use bbox_model or not
    
    Returns
    -------
    predict: predict the state-vector and bounding boxes on each slices 
        multi_state = True: Use the 2D-like tracker (SingleBboxTracker)
        multi_state = False: Use the 3D-like tracker (SliceKalmanFilter)
        predicted bounding boxes are stored in self.predicted_bboxes
    update: add new bounding-boxes to tracks and update by using these.
        estimation: conduct SliceKalmanFilter filtering
        trackers for reid: conduct SingleBboxTracker filtering
    '''
    def __init__(self, id:int, model:SlicedModel, ns:int, bbox_model:BboxModel, obs0:Set[Bbox], multi_state:bool):
        super().__init__(id, model, obs0)
        self.filter: SliceKalmanFilter = SliceKalmanFilter(model=model, obs0=obs0)
        self.multi_state = multi_state
        self.model = model
        self.bbox_model = bbox_model
        self.ns = ns
        # if you use SORT, constract slice-wise SingleBboxTracker's
        if multi_state:
            self.bbox_trackers: Dict[int, SingleBboxTracker] = {}
            for b in obs0:
                self.bbox_trackers[b.slice] = SingleBboxTracker(
                    id=id,
                    model=self.bbox_model,
                    obs0=b
                )
                b.id = self.id
        
    def predict(self):
        '''
        Predict sphere-like state and store predicted bboxes to self.predicted_bboxes 
        
        Returns
        -------
        Array of (x, y, z, r) that is prediction \mu_{t+1|t}
        '''
        # predict each single-bbox-tracker if multi_state is true
        self.predicted_bboxes: Dict[int, np.ndarray] = {s: None for s in range(self.ns)}
        if self.multi_state:
            for s, sbt in self.bbox_trackers.items():
                sbt.predict()
                self.predicted_bboxes[s] = sbt.predicted_bbox # x, P are updated from {t|t} to {t+1|t}
        else:
            # please note that this concludes None
            self.predicted_bboxes = self.filter.observe_predict()
        
        # preidct on estimator (return only mean)
        return self.filter.state_predict()[0].copy()[self.model._primary_idx]
    
    def update(self, bboxes:Set[Bbox]):
        '''
        Update kalman-filter's state by using observations
            multi_state is true: conduct SingleBboxTracker filtering at each slice
            multi_state is false: conduct SliceKalmanFilter filtering
            
        Parameters
        ----------
        bboxes: set of bounding-boxes obtained by re-identification
        '''
        # update each single-bbox-tracker if multi_state is true
        if self.multi_state:
            for b in bboxes:
                if b.slice in self.bbox_trackers.keys():
                    self.bbox_trackers[b.slice].update(obs=b)
                else:
                    self.bbox_trackers[b.slice] = SingleBboxTracker(
                        id=self.id, 
                        model=self.bbox_model,
                        obs0=b
                    )
        
        # update estimator
        self.filter.state_update(y=bboxes)
        
        # assing id to bboxes
        for b in bboxes:
            b.id = self.id
            
        # add to track
        self.tracks.append(bboxes)
        
        # reset staleness because update is occured
        self.staleness = 0

def depth_sort(sliced_bboxes: Dict[int, List[Bbox]], bbox_model:BboxModel, ns:int, max_staleness:int) -> List[Set[Bbox]]:
    '''
    Depth-SORT: Split bounding boxes over slices into ones derived from single object
        conduct vanila SORT as it is to bounding-boxes over slices.
        
    Parameters
    ----------
    sliced_bboxes: bounding boxes sorted by slices
    bbox_model: state-space-model used for depth-directional tracking
    ns: number of slice
    max_staleness: the threshold to determine how much the tracks separated is regarded as the same objects
    
    Returns
    -------
    reidentified bounding boxes: List of bounding-boxes set. This is sorted by ID deriving bounding-boxes
    '''
    
    # make mulit bbox trakcer
    tracker = MultiBboxTrackr(max_staleness, model=bbox_model)
    
    # repeat ns times
    for s in range(ns):
        if s in sliced_bboxes.keys():
            tracker.step(sliced_bboxes[s])
        else:
            tracker.step([])
    return tracker.reidentified_bboxes
    
def unused_dsort_idx(dsort_bboxes: List[Set[Bbox]], identified_bboxs: List[Set[Bbox]]):
    '''
    Return idx of re-identified_bboxes obtained by Depth-SORT whose all bounding-boxes do not used for update
    
    Parameters
    ----------
    dsort_bboxes: re-identified_bboxes obtained by Depth-SORT (sorted by ID)
    identified_bboxes: List of bounding boxes used for update single trackers
    
    Returns 
    -------
    indexes of re-identified_bboxes obtained by Depth-SORT whose all bounding-boxes do not used for update
    '''
    # return dsort_bboxes whose any elements isn't used for all of identified_bboxes
    if len(identified_bboxs) == 0:
        return list(range(len(dsort_bboxes)))
    else:
        return [i for i in range(len(dsort_bboxes)) if max([len(dsort_bboxes[i] & ib) for ib in identified_bboxs]) == 0]