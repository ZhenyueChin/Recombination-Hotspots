
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model
import sys
import pickle
import model
from scipy import stats
def generate_permutations(original):
	'''
	generates all pair permutations of the original matrix
	'''
	attractor_sets=[]
	for i in range(10):
		new = original[:]
		new[i]=new[i]*(-1)

		attractor_sets.append(new)
		for j in range(i+1,10):
			newer = new[:]
			newer[j]=new[j]*(-1)
			attractor_sets.append(newer)
	return attractor_sets

#this is just for generating the video of network dynamics for my thesis defense. Tomorrow morning. Hack time.

with open('networks/original_trials/E5/run1/fitCurve5_2_2_0.pickle', 'rb') as handle:
		  		pop = pickle.load(handle).pop()
		  		if(pop.fitness==1):
		  			print "got one"
		  		toShow = pop
# self,start_state,target_state,max_cycle):


toShow.visualize_network([-1,-1,-1,1,-1,1,1,-1,1,-1],[-1,1,-1,1,-1,-1,1,-1,1,-1],20)