import tensorflow as tf

with tf.gfile.FastGFile(r"C:\Users\hendr\Desktop\python\TrainYourOwnYOLO\model\saved_model.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, ',/',
                         'name.pbtxt', as_text=True)