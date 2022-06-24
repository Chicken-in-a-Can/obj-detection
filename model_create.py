from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader

data = DataLoader.images_dir('hand_pics/')
train_data, test_data = data.split(0.9)

model = object_detector.create(train_data)

loss, accuracy = model.evaluate(test_data)

model.export(export_dir='/home/bob/Downloads/TensorFlow')
