#Part two: Specialization Increases Modularity
#Implemented as simple hillclimber to start, with deterministic perturbations
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model
import sys
import pickle
import time


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
	start_attractorsB = [
						np.array([-1,1,-1,1,-1,1,-1,1,-1,-1]),
						np.array([-1,1,-1,1,-1,1,-1,1,-1,1]),
						np.array([-1,-1,-1,1,-1,1,-1,1,-1,1]),
						np.array([-1,-1,1,1,-1,1,-1,1,-1,1]),
						np.array([-1,-1,1,-1,-1,1,-1,1,-1,1]),
						np.array([-1,-1,1,-1,1,1,-1,1,-1,1]),
						np.array([-1,-1,1,-1,1,-1,-1,1,-1,1]),
						np.array([-1,-1,1,-1,1,-1,1,1,-1,1]),
						np.array([-1,-1,1,-1,1,-1,1,-1,-1,1]),
						np.array([-1,-1,1,-1,1,-1,1,-1,1,1]),
						np.array([-1,-1,1,-1,1,-1,1,-1,1,-1])
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

def evaluate_double(individual, max_cycle, target_attractGirorA, target_attractorB):
	'''
	Run evaluate_single on both target attractors
	'''
	fitness=0
	fitness+=evaluate_single(individual, max_cycle, target_attractorA)
	fitness+=evaluate_single(individual, max_cycle, target_attractorB)
	return fitness/2.0


def tournament(population,contenders,eliminated):
	'''
	given a set of individuals, discard those individuals that are dominated by the set of contenders
	and remove them from the population
	TODO: optimize this function
	'''
	#print len(population)
	for contender in contenders:
		for other in contenders:
			#print contcount,othercount
			if((not other is contender) and other.genetic_age<=contender.genetic_age and other.fitness>=contender.fitness): #contender has been dominated
				#print "individual with age: ",contender.genetic_age," and fitness: ",contender.fitness," dominated by other with age: ",other.genetic_age," and fitness: ",other.fitness
				eliminated.append(contender)
				population.remove(contender)
				return
	

def update_progress(i):
	'''
	updates a timer in the terminal
	'''

	sys.stdout.write("\r%.0f%%" % (i*100))
	sys.stdout.flush()

def entirely_non_dominated(population):
	'''
	checks to see if the set of nondominated individuals is equal in size to the population set
	'''
	return False

def complete(best,population,generations,targetA,max_cycle):

	print "\nPart one complete!"
	if(best.fitness>-1):
		print "The network that produced the most accurate attractors had fitness: " , best.fitness
	print "networks evaluated: " , len(population)*generations+generations 
	# print "Now we apply evolutionary pressure for pattern two"

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

	#best.rectangle_visualization(start_attractors,targetA, "Target A")
	#best.visualize_network(np.array([-1,1,-1,1,-1,1,-1,1,-1,1]),targetA,max_cycle)

	pickle.dump( best, open( "best_network.pickle", "wb" ) )
	#temp = raw_input("Best network saved in best_network.pickle. Press enter to end")
def pareto_visualization(population,eliminated):
	'''
	visualize the pareto front as evolution goes along
	'''
	
	ages_pop=list()
	fitnesses_pop=list()
	ages_elim=list()
	fitnesses_elim=list()
	for individual in population:
		ages_pop.append(individual.genetic_age)
		fitnesses_pop.append(individual.fitness)
	for individual in eliminated:
		ages_elim.append(individual.genetic_age)
		fitnesses_elim.append(individual.fitness)
	#print ages,fitnesses
	plt.cla()

	plt.axis((0,100,0,1.5))
	plt.ylabel('Fitness')
	plt.xlabel('Age')
	plt.plot(ages_pop,fitnesses_pop,"bo")
	plt.plot(ages_elim,fitnesses_elim,"rx")
	plt.draw()
	time.sleep(0.2)

def det_pareto(targetA,targetB, max_cycle, pop_size, generations,mu,p,run_number,num_runs):
	'''
	I will evolve populations of GRNs mirroring the initial results
	"Specialization Increases Modularity" in the original paper, but using an
	Age-Fitenss Pareto Optimization algorithm, and deterministic start attractors
	Returns the Q value of the first network found with fitness 1
	'''
	plt.ion()
	plt.show()
	#initial network: 200 networks with identical randomized edges:
	network_size = len(targetA)
	initial_edges = model.GRN.initialize_edges(network_size,network_size)
	population=list()
	for i in range(pop_size):
		population.append(model.GRN(targetA,max_cycle)) #not identical

	#Find fitness for each individual:
	for individual in population:
		individual.fitness = evaluate_single(individual,max_cycle,targetA)
	#print individual.measure_modularity()
	#evolutionary loop is initiated:
	best = population[0]

	for gen in range(generations):
		
		#each network is evaluated, and mutated
		next_gen = []
		for individual in population:
			individual.genetic_age+=1
			child = individual.copy()
			child.perturb(mu)
			child.fitness = evaluate_single(child,max_cycle,targetA)
			if child.fitness > best.fitness:
				best = child
				#print "new best with fitness: ",best.fitness
			next_gen.append(child)
		population.extend(next_gen)
		
		#one extra random network is added at zero age:
		new_individual = model.GRN(targetA,max_cycle,model.GRN.initialize_edges(network_size,network_size))
		new_individual.fitness = evaluate_single(new_individual,max_cycle,targetA)
		population.append(new_individual)
		if new_individual.fitness > best.fitness:
			best = new_individual
			#print "new best with fitness: ",best.fitness

		#check for termination:
		if(best.fitness==1):
			print "optimal fitness found"
			complete(best,population,gen,targetA,max_cycle)
			break

		#now our population is of size 2k+1, time for tournaments:
		eliminated = []
		while(len(population)>pop_size):
			individualA = random.choice(population)
			individualB = random.choice(population)
			#total_tournament(population,eliminated)
			tournament(population,{individualA,individualB},eliminated)
			
			if(entirely_non_dominated(population)):
				pop_size=len(population)
				print "entirely_non_dominated"
 		#pareto_visualization(population,eliminated)
		#update_progress((gen*1.0*run_number)/((generations-1)*num_runs))
		update_progress((run_number*1.0)/(num_runs))
		#print " Population size: ",len(population)

	return best.measure_modularity()


def test_pareto(run_num,num_runs):
	target = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	max_cycle = 30
	pop_size =50 #target number of nondominated individuals
	generations = 1000
	mu = 0.25
	p=0.15
	return det_pareto(target,target, max_cycle, pop_size, generations,mu,p,run_num,num_runs)


def main():
	#rand.seed("this is a seed") #for safety-harness
	q_values=[]
	trial_counter=0
	with open('seeds.pickle', 'rb') as handle:
		seeds = pickle.load(handle)

	for seed in seeds:
		trial_counter+=1
		rand.seed(seed)
		q_values.append(test_pareto(trial_counter,len(seeds)))

	pickle.dump( q_values, open( "q_values.pickle", "wb" ) )
main()