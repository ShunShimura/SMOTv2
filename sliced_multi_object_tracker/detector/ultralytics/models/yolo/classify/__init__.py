# Ultralytics YOLO 🚀, AGPL-3.0 license

from sliced_multi_object_tracker.detector.ultralytics.models.yolo.classify.predict import ClassificationPredictor
from sliced_multi_object_tracker.detector.ultralytics.models.yolo.classify.train import ClassificationTrainer
from sliced_multi_object_tracker.detector.ultralytics.models.yolo.classify.val import ClassificationValidator

__all__ = "ClassificationPredictor", "ClassificationTrainer", "ClassificationValidator"
