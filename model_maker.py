import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')
dataloader = object_detector.DataLoader.from_pascal_voc("shape_pics", "shape_pics_annotations", label_map={1: "orange circle", 2: "green circle", 3: "green square", 4:"orange square", 5:"none"})
model = object_detector.create(train_data=dataloader, model_spec=spec, epochs=50, batch_size=8, train_whole_model=True)
model.export(export_dir='.')
