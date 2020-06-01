import numpy as np

np.random.seed(0)
#matrices = [np.random.normal(0,1,(40,21)) for x in range(0,100)]
matrices_random = [np.random.normal(0,1,(40,21)) for x in range(0,1000)]
matrices_random = np.array(matrices_random)
binary_labels_random = np.array([np.random.choice([0,1],40,p=[0.5,0.5]) for x in range(0,1000)])


#smaller data
matrices_random_sm = [np.random.normal(0,1,(30,16)) for x in range(0,1000)]
matrices_random_sm = np.array(matrices_random_sm)
binary_labels_random_sm = np.array([np.random.choice([0,1],30,p=[0.5,0.5]) for x in range(0,1000)])

#smallest data
matrices_random_smest = [np.random.normal(0,1,(20,11)) for x in range(0,1000)]
matrices_random_smest = np.array(matrices_random_smest)
binary_labels_random_smest = np.array([np.random.choice([0,1],20,p=[0.5,0.5]) for x in range(0,1000)])
