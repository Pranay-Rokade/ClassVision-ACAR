import torch
import cv2
import numpy as np
from time import time
from ultralytics import YOLO
import os 
import yaml
from easydict import EasyDict as edict
from pathlib import Path

import supervision as sv
from bytetrack.byte_tracker import BYTETracker
from strongsort.strong_sort import StrongSORT

from util import get_config

SAVE_VIDEO = False
TRACKER = "bytetrack"

# class ObjectDetection:
#     def __init__(self, capture_index):

#         self.capture_index = capture_index