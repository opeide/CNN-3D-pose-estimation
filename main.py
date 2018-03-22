__author__ = 'opeide'

from batch_generator import BatchGenerator
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import util
import model
#TODO: Normalize RGB channelse to zero mean and unit variance


tf.logging.set_verbosity(tf.logging.INFO)



dataset_path = 'dataset'
gen = BatchGenerator()
gen.load_dataset(dataset_path)



i=0
for batch, labels in gen.train_input_gen(num_triplets=2):
    i += 1
    if i > 1:
        break
    print(np.shape(batch['x']))


cnn = tf.estimator.Estimator(model_fn=model.cnn_model_fn)#,model_dir="/tmp/logg3")
for i in range(5):
    print("runn nr: ", i)
    cnn.train(input_fn=lambda: next(gen.train_input_gen(num_triplets=2)), steps=10)
    print("Calculate the db space")
    db_space_np = util.get_db_space_np(gen, cnn)
    print("Get histogram array")
    histogram_arrya = util.get_histogram_array(gen, cnn, db_space_np)

    bin = [10, 20, 40, 180]
    print(histogram_arrya)
    util.histogram(histogram_arrya, bin)
