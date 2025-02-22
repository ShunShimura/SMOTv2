# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.34"

from sliced_multi_object_tracker.detector.ultralytics.data.explorer.explorer import Explorer
from sliced_multi_object_tracker.detector.ultralytics.models import RTDETR, SAM, YOLO, YOLOWorld, YOLOv10
from sliced_multi_object_tracker.detector.ultralytics.models.fastsam import FastSAM
from sliced_multi_object_tracker.detector.ultralytics.models.nas import NAS
from sliced_multi_object_tracker.detector.ultralytics.utils import ASSETS, SETTINGS as settings
from sliced_multi_object_tracker.detector.ultralytics.utils.checks import check_yolo as checks
from sliced_multi_object_tracker.detector.ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
    "YOLOv10"
)
