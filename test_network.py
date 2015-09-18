#This file tests a network. Pass in initial configurations, see the results.
#mostly a sanity check
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model
import sys
import pickle


def main(v):
	v = v[1:]
	for el in range(10):
		v[el] = int(v[el])

	nodes = np.array(v)
	max_cycle=20
	with open('best_network.pickle', 'rb') as handle:
  		individual = pickle.load(handle)

	print "the network:" 
	print individual.edges
  	individual.nodes=nodes
	counter = 0
	while(counter <= max_cycle and individual.update_state()):
		counter += 1

	if(counter <= max_cycle):
		print "stable"
		print individual.nodes
	else:
		print chaotic

main(sys.argv)