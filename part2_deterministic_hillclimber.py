#Part two: Specialization Increases Modularity
#Implemented as simple hillclimber to start, with deterministic perturbations
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model
import sys
import pickle


def hamming(attractor, target):
	'''
	TODO: make this cooler
	'''
	count = 0
	for i in range(len(attractor)):
		if(target[i] == attractor[i]):
			count += 1
	return len(attractor)-count


def generate_initial_attractors(target,set_size,p):
	'''
	Generates a random set of perturbations of the target set for evaluation
	'''
	returnables = list()
	possible_values = [-1, 1]
	for i in range(set_size):
		temp = target.copy()
		for j in range(len(temp)):
			if(random.rand()<p):
				temp[j]=np.random.choice(possible_values, 1)[0]
		returnables.append(temp)

	return returnables

def evaluate(individual, max_cycle, target_attractor,p):
	'''
	Run the network until it reaches a stable attractor, or exceeds the allowed number
	of generations, return the fitness of this individual
	'''
	fitness = 0.0
	#generate a DETERMINISTIC set of perturbations of the target attractor:
	#start_attractors = generate_initial_attractors(target_attractor,200,p)
	start_attractors = [
						np.array([-1,1,-1,1,-1,1,-1,1,-1,1]),
						np.array([1,1,-1,1,-1,1,-1,1,-1,1]),
						np.array([1,-1,-1,1,-1,1,-1,1,-1,1]),
						np.array([1,-1,1,1,-1,1,-1,1,-1,1]),
						np.array([1,-1,1,-1,-1,1,-1,1,-1,1]),
						np.array([1,-1,1,-1,1,1,-1,1,-1,1]),
						np.array([1,-1,1,-1,1,-1,-1,1,-1,1]),
						np.array([1,-1,1,-1,1,-1,1,1,-1,1]),
						np.array([1,-1,1,-1,1,-1,1,-1,-1,1]),
						np.array([1,-1,1,-1,1,-1,1,-1,1,1]),
						np.array([1,-1,1,-1,1,-1,1,-1,1,-1])
						]
	fitness_values = list()

	for initial_state in start_attractors:
		#print initial_state
		individual.nodes=initial_state
		#print individual.nodes
		counter = 0
		#print hamming(individual.nodes,target_attractor)
		while(counter <= max_cycle and individual.update_state()):
			counter += 1
			
		if(counter <= max_cycle):
			#print "stable"
			#not chaotic or cyclic
			ham = hamming(individual.nodes,target_attractor)
			#print ham
			this_fitness = (1-(ham/float(len(target_attractor)))) #raise to the 5th
			fitness_values.append(this_fitness)
		else:
			fitness_values.append(0) #zero fitness for chaotic/cyclic state

	print fitness_values
	my_sum = sum(fitness_values)
	#print my_sum
	return my_sum/len(fitness_values)

def update_progress(i):
	'''
	updates a timer in the terminal
	'''

	sys.stdout.write("\r%.0f%%" % (i*100))
	sys.stdout.flush()

def parallel_hill_climber(target, max_cycle, pop_size, generations,mu,p):
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
		population.append(model.GRN(target,initial_edges))


	#Find fitness for each individual:
	for individual in population:
			individual.fitness = evaluate(individual,max_cycle,target,mu)

	#evolutionary loop is initiated:
	best = population[0]
	best.fitness=-1
	for gen in range(generations-1):
		
		#each network is evaluated
		for individual in population:
			#print "nodes: " , individual.nodes
			#print "fitness: ",  evaluate(individual,max_cycle,target)
			child = individual.copy()
			child.perturb(mu)
			child.fitness = evaluate(child,max_cycle,target,p)
			#print "child fitness: " , child.fitness
			if child.fitness > individual.fitness:
				# print child.fitness, " better than: " , individual.fitness

				individual = child

				if individual.fitness > best.fitness:
					best = individual
					print "new best with fitness: " , best.fitness
					

		
		update_progress(gen*1.0/(generations-1))
	print "\nComplete!"
	if(best.fitness>-1):
		print "The network that produced the most accurate attractors had fitness: " , best.fitness
	print "networks evaluated: " , len(population)*generations

	with open('best_network.pickle', 'wb') as handle:
 		pickle.dump(best, handle)
	

def test_hill_climber():
	target        = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	max_cycle = 20
	pop_size = 1 #parallel climbers
	generations = 50
	mu = 0.05
	p=0.15
	parallel_hill_climber(target, max_cycle, pop_size, generations,mu,p)


def main():

	test_hill_climber()

main()