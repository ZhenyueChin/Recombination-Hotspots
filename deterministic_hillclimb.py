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
			if(rand.rand()<p):
				temp[j]=np.rand.choice(possible_values, 1)[0]
		returnables.append(temp)

	return returnables

def evaluate_single(individual, max_cycle, target_attractor):
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
		individual.nodes=np.zeros(individual.nodes.shape)
		individual.nodes[0]=initial_state
		counter = 1
		while(counter < max_cycle and individual.update_state(counter)):
			counter += 1

		if(counter < max_cycle):
			#stable, not chaotic or cyclic
			ham = hamming(individual.nodes[counter-1],target_attractor)
			#print individual.nodes[counter-1]
			this_fitness = (1-(ham/float(len(target_attractor)))) #raise to the 5th
			fitness_values.append(this_fitness)
		else:
			fitness_values.append(0) #zero fitness for chaotic/cyclic state

	#print fitness_values
	my_sum = sum(fitness_values)

	return my_sum/len(fitness_values)

def evaluate_double(individual, max_cycle, target_attractorA, target_attractorB):
	'''
	Run evaluate_single on both target attractors
	'''
	fitness=0
	fitness+=evaluate_single(individual, max_cycle, target_attractorA)
	fitness+=evaluate_single(individual, max_cycle, target_attractorB)
	return fitness/2.0

def update_progress(i):
	'''
	updates a timer in the terminal
	'''

	sys.stdout.write("\r%.0f%%" % (i*100))
	sys.stdout.flush()

def det_hillclimb(targetA,targetB, max_cycle, pop_size, generations,mu,p):
	'''
	As a very simple first step, I will evolve populations of GRNs mirroring the initial results
	"Specialization Increases Modularity" in the original paper, but using a hill climber evolutionary
	algorithm
	'''

	#initial network: 200 networks with identical randomized edges:
	network_size = len(targetA)
	initial_edges = model.GRN.initialize_edges(network_size,network_size)
	population=list()
	for i in range(pop_size):
		population.append(model.GRN(targetA,max_cycle,initial_edges))

	#Find fitness for each individual:
	for individual in population:
		individual.fitness = evaluate_single(individual,max_cycle,targetA)
	print "initial fitness of network: "+str(individual.fitness)

	#evolutionary loop is initiated:
	best = population[0]
	#best.fitness=-1
	for gen in range(generations):
		
		#each network is evaluated
		for individual in range(len(population)):
			#print "nodes: " , individual.nodes
			#print "fitness: ",  evaluate(individual,max_cycle,target)
			child = population[individual].copy()
			child.perturb(mu)
			child.fitness = evaluate_single(child,max_cycle,targetA)
			#print "child fitness: " , child.fitness
			if child.fitness > population[individual].fitness:
				# print child.fitness, " better than: " , individual.fitness

				population[individual] = child

				if population[individual].fitness > best.fitness:
					best = population[individual]
				if(pop_size==1):
					print "new best with fitness: " , best.fitness
				else:
					temp_fits = list()
					for ind in population:
						temp_fits.append(("%.2f" % ind.fitness))

					sys.stdout.write('\r')
					sys.stdout.write('    '+str(temp_fits))
		update_progress(gen*1.0/(generations-1))
	print "\nPart one complete!"
	if(best.fitness>-1):
		print "The network that produced the most accurate attractors had fitness: " , best.fitness
	print "networks evaluated: " , len(population)*generations 
	# print "Now we apply evolutionary pressure for pattern two"


	# #Find fitness for each individual:
	# for individual in population:
	# 	individual.fitness = evaluate_double(individual,max_cycle,targetA,targetB)
	# best=population[0]
	# for gen in range(generations):
		
	# 	#each network is evaluated
	# 	for individual in range(len(population)):
	# 		#print "nodes: " , individual.nodes
	# 		#print "fitness: ",  evaluate(individual,max_cycle,target)
	# 		child = population[individual].copy()
	# 		child.perturb(mu)
	# 		child.fitness = evaluate_double(child,max_cycle,targetA,targetB)
	# 		#print "child fitness: " , child.fitness
	# 		if child.fitness > population[individual].fitness:
	# 			# print child.fitness, " better than: " , individual.fitness

	# 			population[individual] = child

	# 			if population[individual].fitness > best.fitness:
	# 				best = population[individual]
	# 			if(pop_size==1):
	# 				print "new best with fitness: " , best.fitness
	# 			else:
	# 				temp_fits = list()
	# 				for ind in population:
	# 					temp_fits.append(("%.2f" % ind.fitness))

	# 				sys.stdout.write('\r')
	# 				sys.stdout.write('    '+str(temp_fits))

	# 	update_progress(gen*1.0/(generations-1))
	# print "\nPart two complete!"
	# if(best.fitness>-1):
	# 	print "The network that produced the most accurate attractors had fitness: " , best.fitness
	# with open('best_network.pickle', 'wb') as handle:
 # 		pickle.dump(best, handle)
 # 	print "networks evaluated: " , len(population)*generations 
 # 	print "Best network saved in best_network.pickle for further study"
	
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

	best.rectangle_visualization(start_attractors,targetA, "Target A")
	best.visualize_network(np.array([1,-1,1,-1,-1,1,-1,1,-1,1]),targetA,max_cycle)
	temp = raw_input("enter to end")











def test_hill_climber():
	targetA        = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	targetB        = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	max_cycle = 20
	pop_size = 2 #parallel climbers
	generations_per_pattern = 10
	mu = 0.05
	p=0.15
	det_hillclimb(targetA, targetB, max_cycle, pop_size, generations_per_pattern,mu,p)


def main():
	#rand.seed("hppufaejfpaoiwejfilwjef;iljfw") #for safety-harness
	test_hill_climber()

main()