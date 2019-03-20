import glob
import os
import math
import numpy as np
import cv2

def convert_to_one_hot(Y, C):
	Y = np.eye(C)[Y.reshape(-1)].T
	print("here2: ", Y.shape)
	return Y

def load_data():
	data = np.load('hyf_data.npy')
	data = data[np.random.permutation(len(data))]
	print(data.shape)
	x_img        = []
	y_class      = []
	y_coord      = []
	y_visibility = []
	y_pose       = []
	y_gender     = []

	for images in data:
		images[0] = cv2.resize(images[0],(227,227))
		x_img.append(images[0])
		y_class.append(images[1][0])
		y_coord.append(images[2])
		y_visibility.append(images[3])
		y_pose.append(images[4])
		y_gender.append(images[5])

	X        = np.array(x_img)
	y_class      = np.array(y_class)   
	y_coord      = np.array(y_coord)   
	y_visibility = np.array(y_visibility)   
	y_pose       = np.array(y_pose)   
	y_gender     = np.array(y_gender)   

	X = X/255.
	Y = convert_to_one_hot(y_class, 2).T

	print("shape of input data: ", X.shape)
	print("shape of target variable: ", Y.shape)
	print("shape of y_coord: ", y_coord.shape)
	print("shape of y_visibility: ", y_visibility.shape)
	print("shape of y_pose: ", y_pose.shape)
	print("shape of y_gender: ", y_gender.shape)
	

	return X, Y, y_coord, y_visibility, y_pose, y_gender

load_data()

def random_mini_batches(X, Y, gen, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]              # number of training examples
    print("m: ", m)
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    #print("shuffled_X: ",shuffled_X.shape)
    #print("Y_shape: ", Y.shape)
    #print("perm: ", max(perm))
    #print("y_perm: ", Y[permutation,:].shape)
    shuffled_Y = Y[permutation,:]
    shuffled_g = gen[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_g = shuffled_g[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_g)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_g = shuffled_g[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_g)
        mini_batches.append(mini_batch)
    
    return mini_batches

