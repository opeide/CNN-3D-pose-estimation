from batch_generator import BatchGenerator
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import util
import model
#TODO: Normalize RGB channelse to zero mean and unit variance









dataset_path = 'dataset'
gen = BatchGenerator()
gen.load_dataset(dataset_path)

triplet_sequence = gen.all_triplets()
util.plot_triplet_sequence(triplet_sequence)


cnn = tf.estimator.Estimator(model_fn=model.cnn_model_fn)



cnn.train(input_fn=model.train_input_fn)
