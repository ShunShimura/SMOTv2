import os,sys,shutil

from typing import Set, Tuple
import numpy as np

from sliced_multi_object_tracker.core import *
from sliced_multi_object_tracker.filter import *
from sliced_multi_object_tracker.tracker.tracker import *

EPS = 1e-6

class SingleBboxTracker(BaseSingleObjectTracker):
    '''
    Single Object Tracker regarding the state-vector as (x, y, w, h) and their differentiations
    
    Attributes
    ----------
    id: number to identify
    model: state-space-model 
    obs0: initial observation
    filter: kalman filter to estimate 2D-like state and predict bounding box
    tracks: past tracks of bounding box at each time
    predicted_bbox: predictions of bounding box
    
    Methods
    -------
    predict: conduct KalmanPredict and predict bounding box (stored self.predicted_bbox)
    update: conduct KalmanFiltering 
    '''
    def __init__(self, id:int, model:BboxModel, obs0:Bbox):
        super().__init__(id, model, obs0)
        self.model = model
        self.filter = KalmanFilter(
            obs0=obs0.bbox,
            model=model
        )

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Predict state-vector {t+1|t} and store predicted observations in self.predicted_bbox
        
        Returns
        -------
        x: predicted state-vector \mu_{t+1|t}
        P: predicted state-vector error covariance P_{t+1|t}
        '''
        # predict x, P on {t+1|t}
        self.filter.state_predict()
        
        # preserve observation prediction
        self.predicted_bbox = self.filter.observe_predict()
        
        return self.filter.x.copy(), self.filter.P.copy()
    
    def update(self, obs:Bbox):
        # filter update
        self.filter.state_update(obs.bbox)
        
        # add to track
        self.tracks.append(obs)
        
        # give id to bbox
        if self.id is not None:
            obs.id = self.id
        
        # reset staleness
        self.staleness = 0

class MultiBboxTrackr(BaseMultiObjectTracker):
    '''
    This class conducts SORT (inputs are bounding-boxes from single image)
    
    Attributes
    ----------
    single_trackers: each object's tracker
    id_counter: use it to get new id on generation
    max_staleness: How much regard the tracks separated as an identical track
    model: the state-space-model used for tracking
    reidentified_bboxes: the result obtained by this class, which is list of bounding-boxes sored by ID
    
    Methods
    -------
    step: Main process of tracking; compare prediction from past tracks and observations and assing ID according to the matching
    _save: update self.reidentified_bboxes
    active_trackers: return the single_trackers whose staleness is less than max_staleness
    '''
    def __init__(self, max_staleness:int, model:BboxModel):
        super().__init__()
        self.max_staleness = max_staleness
        self.model = model
        self.single_trackers: List[SingleBboxTracker] = []
        self.reidentified_bboxes: List[Set[Bbox]] = []
        
    def step(self, bboxes: List[Bbox]) -> None:
        '''
        Assing ID to newly obtained bounding boxes and update single trackers
            1. matching between predictions and observations
            2. update single trackers, stale non-updated trackers and generated new tracker using non-assined bounding box
            3. calculate KalmanPredict and predict bounding box
            
        Parameters
        ----------
        bboxes: List of bounding boxes corresponding one slice 
        '''
        # matching
        active_trackers: List[SingleBboxTracker] = self.active_trackers()
        predicted_bboxes = [sbt.predicted_bbox for sbt in active_trackers]
        observed_bboxes = [b.bbox for b in bboxes]
        matched_idx, stale_idx, new_idx = hungarian_matching(
            predicted_bboxes,
            observed_bboxes,
            f=intersection_of_union,
            threshold=EPS
        )
        
        # update
        for i, j in matched_idx:
            active_trackers[i].update(bboxes[j])
            
        # generate
        for j in new_idx:
            self.single_trackers.append(
                SingleBboxTracker(id=None, model=self.model, obs0=bboxes[j])
            )
            
        # stale
        for i in stale_idx:
            active_trackers[i].stale()
        
        # predict
        for at in self.active_trackers():
            at.predict()
            
        # save
        self._save()
    
    def _save(self) -> List[Set[Bbox]]:
        '''
        Save re-identified bboxes of list
        
        Returns
        -------
        reidentified_bboxes: bounding_boxes sorted by ID deriving them
        '''
        self.reidentified_bboxes = [
            set(sot.tracks) for sot in self.single_trackers
        ]
        return self.reidentified_bboxes
    
    def active_trackers(self) -> List[SingleBboxTracker]:
        '''
        Return tracking single trackers
        
        Returns
        -------
        List of single trackers whose staleness is less than max_staleness
        '''
        return [sot for sot in self.single_trackers if not sot.is_stale(self.max_staleness)]