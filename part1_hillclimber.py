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
	return len(attractor)-count


def generate_initial_attractors(target,set_size,mu):
	'''
	Generates a random set of perturbations of the target set for evaluation
	'''
	returnables = list()
	possible_values = [-1, - 1]
	for i in range(set_size):
		temp = target.copy()
		for j in range(len(temp)):
			if(random.rand()<mu):
				temp[j]=np.random.choice(possible_values, 1)[0]
		returnables.append(temp)

	print returnables
	return returnables

def evaluate(individual, max_cycle, target_attractor,mu):
	'''
	Run the network until it reaches a stable attractor, or exceeds the allowed number
	of generations, return the fitness of this individual
	'''
	fitness = 0
	#generate a random set of perturbations of the target attractor:
	start_attractors = generate_initial_attractors(target_attractor,200,mu)
	fitness_values = list()

	for initial_state in start_attractors:
		individual.nodes=initial_state
		counter = 0
		while(counter <= max_cycle and individual.update_state()):
			counter += 1

		if(counter <= max_cycle):
			#not chaotic or cyclic
			ham = hamming(individual.nodes,target_attractor)
			this_fitness = (1-(ham/float(len(target_attractor)))) #raise to the 5th
			fitness_values.append(this_fitness)
		else:
			fitness_values.append(0) #zero fitness for chaotic/cyclic state

	print fitness_values
	my_sum = sum(fitness_values)
	print my_sum
	return my_sum/len(fitness_values)

def update_progress(i):
	'''
	updates a timer in the terminal
	'''

	sys.stdout.write("\r%.0f%%" % (i*100))
	sys.stdout.flush()

def parallel_hill_climber(target, initial_nodes, max_cycle, pop_size, generations,mu):
	'''
	As a very simple first step, I will evolve populations of GRNs mirroring the initial results
	"Specialization Increases Modularity" in the original paper, but using a hill climber evolutionary
	algorithm
	'''

	#initial network: 200 networks with identical randomized edges:
	network_size = len(target)
	initial_edges = model.GRN.initialize_edges(network_size,network_size)
	population=list()
	for i in range(pop_size):
		population.append(model.GRN(initial_nodes,initial_edges))


	#First we test for chaos in our initial seed population. Ask Bongard about this.
	counter = 0
	while(counter < max_cycle and population[0].update_state()):
		counter += 1
	
	print "initial cycle length: ",counter
	if(counter==max_cycle):
		print "initial network was chaotic: restart?"
		#return

	#Find fitness for each individual:
	for individual in population:
			individual.fitness = evaluate(individual,max_cycle,target,mu)

	#evolutionary loop is initiated:
	best = population[0]
	best.fitness=-1
	for gen in range(generations-1):
		
		#each network is evaluated
		for individual in population:
			individual.nodes=initial_nodes
			#print "nodes: " , individual.nodes
			#print "fitness: ",  evaluate(individual,max_cycle,target)
			child = individual.copy()
			child.perturb(mu)
			child.fitness = evaluate(child,max_cycle,target,mu)
			#print "child fitness: " , child.fitness
			if child.fitness > individual.fitness:
				# print child.fitness, " better than: " , individual.fitness

				individual = child

				if individual.fitness > best.fitness:
					best = individual
					print "new best: " , best.nodes , " with fitness: " , best.fitness
					

		
		update_progress(gen*1.0/(generations-1))
	print "\nComplete!"
	if(best.fitness>-1):
		print " here is the closest stable attractor to target: " , best.nodes , " with fitness: " , best.fitness
	global NUM_CHAOTIC_NETWORKS
	print "chaotic networks: ", NUM_CHAOTIC_NETWORKS
	print "networks evaluated: " , len(population)*generations


def test_hill_climber():
	target        = np.array([-1,-1,-1,-1,1,1,1,1,-1,-1])
	initial_nodes = np.array([1,-1,1,1,-1,-1,1,-1,1,1])
	max_cycle = 20
	pop_size = 1 #parallel climbers
	generations = 50000
	mu = 0.05
	parallel_hill_climber(target, initial_nodes, max_cycle, pop_size, generations,mu)


def main():

	test_hill_climber()

main()