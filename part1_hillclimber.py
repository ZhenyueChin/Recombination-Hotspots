#Part one: Specialization Increases Modularity
#Implemented as simple hillclimber to start
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model
import sys
NUM_CHAOTIC_NETWORKS = 0
def hamming(attractor, target):
	'''
	TODO: make this cooler
	'''
	count = 0
	for i in range(len(attractor)):
		if(target[i] == attractor[i]):
			count += 1
	return len(attractor)-count

def fitness(attractor,target):
	'''
	Determine the fitness of a network after it reaches a stable attractor,
	using Hamming distance, as described in page 8 of the original paper.
	'''

	return (1-(hamming(attractor,target)/float(len(target)))) #raise to the 5th

def evaluate(individual, max_cycle, target_attractor):
	'''
	Run the network until it reaches a stable attractor, or exceeds the allowed number
	of generations, return the fitness of this individual
	'''
	counter = 0
	while(counter <= max_cycle and individual.update_state()):
		counter += 1

	if(counter > max_cycle):
		#print "chaotic"
		global NUM_CHAOTIC_NETWORKS
		NUM_CHAOTIC_NETWORKS+=1
		return -1
	return fitness(individual.nodes,target_attractor)

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
			individual.fitness = evaluate(individual,max_cycle,target)

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
			child.fitness = evaluate(child,max_cycle,target)
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

def test_chaos():
	pop_size=5000
	max_cycle=500
	initial_nodes = np.array([1,1,1,1,1,1,1,1,1,1])
	
	count_hash = dict()
	population=list()
	for i in range(pop_size):
		initial_edges = model.GRN.initialize_edges(10,10)
		grn = (model.GRN(initial_nodes,initial_edges))

		counter = 0
		while(counter < max_cycle and grn.update_state()):
			counter += 1
	
		if counter in count_hash:
			count_hash[counter]+=1
		else:
			count_hash[counter]=1
		update_progress(float(i)/pop_size)
	
	for key, value in sorted(count_hash.items()):
		print("{} : {}".format(key, value))


def test_hill_climber():
	target        = np.array([-1,-1,-1,-1,1,1,1,1,-1,-1])
	initial_nodes = np.array([1,-1,1,1,-1,-1,1,-1,1,1])
	max_cycle = 20
	pop_size = 1 #parallel climbers
	generations = 50000
	mu = 0.05
	parallel_hill_climber(target, initial_nodes, max_cycle, pop_size, generations,mu)


def main():
	
	test_chaos()
	# test_hill_climber()

main()