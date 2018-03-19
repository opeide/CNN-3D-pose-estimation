from batch_generator import BatchGenerator
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import util
import model
#TODO: Normalize RGB channelse to zero mean and unit variance


tf.logging.set_verbosity(tf.logging.INFO)



dataset_path = 'dataset'
gen = BatchGenerator()
gen.load_dataset(dataset_path)

# i=0
# for trip_batch in gen.train_input_fn(batch_size=1):
#     i += 1
#     if i > 1:
#         break
#     print(trip_batch[2,32,32])



cnn = tf.estimator.Estimator(model_fn=model.cnn_model_fn)
cnn.train(input_fn=lambda: next(gen.train_input_gen(batch_size=1)), steps=101)
