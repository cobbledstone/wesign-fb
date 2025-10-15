import numpy as np

# Load the .npy file
array = np.load('dataset/Hello/sample0.npy')

# Now you can use the array
print(len(array[0]))