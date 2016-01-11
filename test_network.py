#This file tests a network. Pass in initial configurations, see the results.
#mostly a sanity check
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model
import sys
import pickle
import model


def main(v):
	# v = v[1:]
	# for el in range(10):
	# 	v[el] = int(v[el])

	# nodes = np.array(v)
	# max_cycle=20
	
	# with open('best_network.pickle', 'rb') as handle:
 #  		individual = pickle.load(handle)

	# print "the network:" 
	# print individual.edges

	
 #  	individual.nodes=nodes
 #  	print individual.nodes
 #  	individual.visualize_network()
	# counter = 0
	# while(counter <= max_cycle and individual.update_state()):
	# 	counter += 1
	# 	print individual.nodes
	# 	individual.visualize_network()

	# if(counter <= max_cycle):
	# 	print "stable"
	# 	print individual.nodes
	# else:
	# 	print "chaotic or cyclic"

	# individual.visualize_network()
	max_cycle=20
	with open('networks/E1/populationsA.pickle', 'rb') as handle:
  		populationsA = pickle.load(handle)

  	
	attractor_sets = [[ [-1,1,-1,1,-1,1,-1,1,-1,1],
						[1,1,-1,1,-1,1,-1,1,-1,1],
						[-1,-1,-1,1,-1,1,-1,1,-1,1],
						[-1,1,1,1,-1,1,-1,1,-1,1],
						[-1,1,-1,-1,-1,1,-1,1,-1,1],
						[-1,1,-1,1,1,1,-1,1,-1,1],
						[-1,1,-1,1,-1,-1,-1,1,-1,1],
						[-1,1,-1,1,-1,1,1,1,-1,1],
						[-1,1,-1,1,-1,1,-1,-1,-1,1],
						[-1,1,-1,1,-1,1,-1,1,1,1],
						[-1,1,-1,1,-1,1,-1,1,-1,-1]
						],
					  [[-1,1,-1,1,-1,-1,1,-1,1,-1],
					  [1,1,-1,1,-1,-1,1,-1,1,-1],
					  [-1,-1,-1,1,-1,-1,1,-1,1,-1],
					  [-1,1,1,1,-1,-1,1,-1,1,-1],
					  [-1,1,-1,-1,-1,-1,1,-1,1,-1],
					  [-1,1,-1,1,1,-1,1,-1,1,-1],
					  [-1,1,-1,1,-1,1,1,-1,1,-1],
					  [-1,1,-1,1,-1,-1,-1,-1,1,-1],
					  [-1,1,-1,1,-1,-1,1,1,1,-1],
					  [-1,1,-1,1,-1,-1,1,-1,-1,-1],
					  [-1,1,-1,1,-1,-1,1,-1,1,1]
					  ]]
	target_state = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	for individual in populationsA[0]:
  		print model.GRN.evaluate_network(individual, max_cycle, 2 ,attractor_sets)
  	individual=populationsA[0][0]

	individual.rectangle_visualization(attractor_sets[0],target_state, "TargetA")

	for start_state in attractor_sets[0]:
  		individual.visualize_network(start_state,target_state,max_cycle)
  		temp = raw_input("enter to end")
main(sys.argv)