import os,sys,shutil
from typing import List

from sliced_multi_object_tracker.core import *
from sliced_multi_object_tracker.filter import *

class IdCounter:
    '''
    Used for generating the object on tracking.
    '''
    def __init__(self):
        self.new_id = 0
    
    def get_id(self):
        self.new_id += 1
        return int(self.new_id - 1)
    
class BaseMultiObjectTracker:
    '''
    Base class for MultiObjectTracker
        constract vacant list for each object's tracker, and ID generator.
        step is the main process for tracking
        _save is used in step to save interim results
    '''
    def __init__(self):
        self.single_trackers: List[BaseSingleObjectTracker] = []
        self.id_counter = IdCounter()
    
    def step(self):
        return NotImplementedError
    
    def _save(self):
        return NotImplementedError
    
class BaseSingleObjectTracker:
    '''
    Base class for each object's trackeer
    
    Attributes
    ----------
    id: the number to identify
    staleness: how much the update is not done
    tracks: the list of past tracks (consists of observations)
    
    Methods
    -------
    update: add informations in each time process
    is_stale: whether the tracking is terminated or not
    stale: increase staleness 
    '''
    def __init__(self, id:int, model:BaseModel, obs0):
        self.id:int = id
        self.staleness:int = 0
        self.tracks:List = [obs0]
        
    def update(self, obs):
        return NotImplementedError
    
    def is_stale(self, threshold:int):
        return self.staleness > threshold
    
    def stale(self):
        self.staleness += 1