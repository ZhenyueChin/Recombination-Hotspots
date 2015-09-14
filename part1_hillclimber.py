#Part one: Specialization Increases Modularity
#Implemented as simple hillclimber to start
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model

def fitness(individual):
	'''
	Determine the fitness of a network after it reaches a stable attractor,
	using Hamming distance, as described in page 8 of the original paper
	'''

def evaluate(individual,max_generations):
	'''
	Run the network until it reaches a stable attractor, or exceeds the allowed number
	of generations, return the fitness of this individual
	'''
	counter = 0
	while(counter <= max_generations && individual.update_state()):
		counter += 1
		print individual.nodes
	return fitness(individual)

def hill_climber():
	'''
	As a very simple first step, I will evolve populations of GRNs mirroring the initial results
	"Specialization Increases Modularity" in the original paper, but using a hill climber evolutionary
	algorithm
	'''

	#initial network state: 200 networks with identical randomized edges:
	edges = model.GRN.initialize_edges(10,10)
	nodes = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	population=list()
	for i in range(200):
		population.append(model.GRN(nodes,edges))

	#








def main():
	hill_climber()
main()