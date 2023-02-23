import tensorflow as tf

from tensorflow.keras.layers import Input
from src.keras_yolo3.yolo3.model import yolo_body
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


keras_model_path = r"C:\Users\hendr\Desktop\python\TrainYourOwnYOLO\2_Training\src\keras_yolo3\yolo.h5"
keras_weight_path = r"C:\Users\hendr\Desktop\python\TrainYourOwnYOLO\Data\Model_Weights\trained_weights_final.h5"

saved_model_path = r"model"
# model = tf.keras.models.load_model(keras_model_path)

num_anchors = 9
num_classes = 1

model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)

model.load_weights(keras_weight_path)
model.save(saved_model_path)

with tf.gfile.FastGFile(r"C:\Users\hendr\Desktop\python\TrainYourOwnYOLO\model.pb\saved_model.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, ',/',
                         'name.pbtxt', as_text=True)