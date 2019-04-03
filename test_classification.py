import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
from PIL import Image
from scipy import ndimage
from load_data import load_dataset
from logistic_regression import LogisticRegression
from draw_graph import drawGraph

train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()


number_of_epochs = 2000
LR = LogisticRegression(max_number_of_iterations=number_of_epochs, learning_rate=0.005, verbose=True)
LR.fit(train_set_x,train_set_y)
training_loss = LR.costs_
print "Training Loss = %f" %(training_loss[-1])
training_accuracy = LR.predict(train_set_x,train_set_y)
print "Training Accuracy = %f" %(training_accuracy) +" %"
testing_accuracy = LR.predict(test_set_x, test_set_y)
print "Testing Accuracy = %f" %(testing_accuracy) +" %"
drawGraph(number_of_epochs, training_loss, training_accuracy, testing_accuracy)
