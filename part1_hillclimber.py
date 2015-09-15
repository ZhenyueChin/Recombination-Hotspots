#Part one: Specialization Increases Modularity
#Implemented as simple hillclimber to start
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model
import sys

def hamming(attractor, target):
	'''
	TODO: make this cooler
	'''
	count = 0
	for i in range(len(attractor)):
		if(target[i] == attractor[i]):
			count += 1
	return count

def fitness(attractor,target):
	'''
	Determine the fitness of a network after it reaches a stable attractor,
	using Hamming distance, as described in page 8 of the original paper.
	'''

	return (1-hamming(attractor,target))/len(target)

def evaluate(individual, max_generations, target_attractor):
	'''
	Run the network until it reaches a stable attractor, or exceeds the allowed number
	of generations, return the fitness of this individual
	'''
	counter = 0
	stabilized = False
	while(counter <= max_generations and not stabilized):
		stabilized = individual.update_state()
		counter += 1
		#print individual.nodes
	return hamming(individual.nodes,target_attractor)

def update_progress(i):
	'''
	updates a timer in the terminal
	'''

	sys.stdout.write("\r%d%%" % i*100)
	sys.stdout.flush()

def parallel_hill_climber(target, max_cycle, population, generations,mu):
	'''
	As a very simple first step, I will evolve populations of GRNs mirroring the initial results
	"Specialization Increases Modularity" in the original paper, but using a hill climber evolutionary
	algorithm
	'''

	#initial network: 200 networks with identical randomized edges:
	network_size = len(target)
	edges = model.GRN.initialize_edges(network_size,network_size)
	population=list()
	for i in range(population):
		population.append(model.GRN(network_size,edges))

	#each network is perturbed appropriately
	for individual in population:
		individual.perturb(0.15)

	best = population[0]
	#evolutionary loop is initiated:
	for gen in range(generations):
		update_progress(gen/(generations*1.0))

		#each network is evaluated
		for individual in population:

			individual.fitness = evaluate(individual,max_cycle,target)
			child = individual.copy()
			child.perturb(mu)
			child.fitness = evaluate(child,max_cycle,target)

			if child.fitness > individual.fitness:
				individual = child

			if individual.fitness > best:
				print "new best: " + best.nodes


	








def main():
	parallel_hill_climber([0,0,0,0,0,0,0,0,0,0],100,50,50,0.05)
main()