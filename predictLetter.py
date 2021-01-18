import glob
import sys

import cv2
import numpy as np
import os
import tensorflow as tf
import keras
load_model = keras.models.load_model


def get_inference_vector_one_frame_alphabet(image, model):

    video_names = []    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape(1, 200, 200, 1)

    results = model.predict(img_arr)
    results = np.squeeze(results)
       
    return results
    

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def load_label_dicts(label_file):
    id_to_labels = load_labels(label_file)
    labels_to_id = {}
    i = 0

    for id in id_to_labels:
        labels_to_id[id] = i
        i += 1

    return id_to_labels, labels_to_id
    
    
    
def predict(image):
    label_file = 'output_labels_alphabet.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)
    model = load_model('cnn_model.h5')

    prediction_vector = get_inference_vector_one_frame_alphabet(image, model)

    predicted = id_to_labels[prediction_vector.argmax()]
    return predicted
