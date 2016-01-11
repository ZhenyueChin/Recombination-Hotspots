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
	with open('networks/E2/populationsA.pickle', 'rb') as handle:
  		populationsA = pickle.load(handle)
  	with open('networks/E2/populationsB.pickle', 'rb') as handle:
  		populationsB = pickle.load(handle)


  	
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
	#for targetA
	# target_stateA = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	fitnessesA = [model.GRN.evaluate_network(individual, max_cycle, 1 ,attractor_sets) for individual in populationsA[0]]
	percentile95A=np.percentile(fitnessesA, 95)
	top_performersA = [individual for individual in populationsA[0] if individual.fitness>=percentile95A]
	print top_performersA
	print "avg fitness for 95th percentile targetA: "+str(np.mean([i.fitness for i in top_performersA]))
	print "avg mod for 95th percentile targetA: "+str(np.mean([i.measure_modularity() for i in top_performersA]))

	#for targetB
	# target_stateB = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	fitnessesB = [model.GRN.evaluate_network(individual, max_cycle, 2 ,attractor_sets) for individual in populationsB[0]]
	percentile95B=np.percentile(fitnessesB, 95)
	top_performersB = [individual for individual in populationsB[0] if individual.fitness>=percentile95B]
	print top_performersB
	print "avg fitness for 95th percentile targetB: "+str(np.mean([i.fitness for i in top_performersB]))
	print "avg mod for 95th percentile targetB: "+str(np.mean([i.measure_modularity() for i in top_performersB]))

  	individual=top_performersA[0]
	target_stateA = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	individual.rectangle_visualization(attractor_sets[0],target_stateA, "TargetA")

	for start_state in attractor_sets[0]:
  		individual.visualize_network(start_state,target_state,max_cycle)
  		temp = raw_input("enter to end")
main(sys.argv)