__author__ = 'opeide'

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

def sequence_to_triplets(arr):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(arr), 3):
        yield arr[i:i + 3]

def plot_triplet_sequence(triplet_sequence):
    for triplet in sequence_to_triplets(triplet_sequence):
        fig = plt.figure()
        for i in range(3):
            fig.add_subplot(1, 3, i + 1)
            img = plt.imread(triplet[i])
            plt.imshow(img)
        plt.show()

def histogram(arr,bin):
    plt.hist(arr, bin)
    plt.title("Histogram of angels")
    plt.show()

def histogram_generation():
    return [1, 1, 2, 2, 3]

def quternion_angel(q1, q2):
    return 2*np.arccos(np.fabs(np.dot(q1, q2)))*180/np.pi

def loaded_normalized_img(path):
    loaded_img = cv2.imread(path)
    loaded_img_min = loaded_img.min(axis=(0, 1), keepdims=True)
    loaded_img_max = loaded_img.max(axis=(0, 1), keepdims=True)
    return(loaded_img - loaded_img_min) / (loaded_img_max - loaded_img_min)

def get_db_space_np(gen, cnn):
    db_data = gen.gen_db()
    db_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": db_data}, num_epochs=1, shuffle=False)
    db_features = cnn.predict(input_fn=db_input_fn)
    j = 0
    db_space = []
    for db_feature in db_features:
        db_space.append(db_feature["descriptor"])
        j += 1
        if j > 15:
            break
    db_space_np = np.reshape(db_space, [-1, 16])
    return db_space_np

def get_histogram_array(gen, cnn, db_space_np):
    bf = cv2.BFMatcher()
    histogram_arry = []
    for test, classification, quternion in gen.test_gen():
        test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test}, num_epochs=1, shuffle=False)
        test_features = cnn.predict(input_fn=test_input_fn)
        for test_feature in test_features:
            matches = bf.match(db_space_np, np.reshape(test_feature["descriptor"], [-1, 16]))
            matches = sorted(matches, key=lambda x: x.distance)
            db_classificaion, db_quaterion = gen.get_classification_and_quaternion_db(matches[0].queryIdx)
            if classification == db_classificaion:
                angel = quternion_angel(quternion, db_quaterion)
                histogram_arry.append(angel)
    return histogram_arry
