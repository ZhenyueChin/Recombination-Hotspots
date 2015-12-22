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

#@profile
def evaluate_network(individual, max_cycle, num_attractor_sets,attractor_sets):
	'''
	Run the network until it reaches a stable attractor, or exceeds the allowed number
	of generations, return the fitness of this individual
	'''
	fitness = 0.0
	#generate a DETERMINISTIC set of perturbations of the target attractor:
	#start_attractors = generate_initial_attractors(target_attractor,200,p)
	fitness_values = list()
	target_attractors=[np.array([-1,1,-1,1,-1,1,-1,1,-1,1]),
					   np.array([-1,1,-1,1,-1,-1,1,-1,1,-1])]

	for set_index in range(num_attractor_sets):
	
		for initial_state in attractor_sets[set_index]:
			individual.nodes=np.zeros(individual.nodes.shape)
			individual.nodes[0]=initial_state
			counter = 1
			while(counter < max_cycle and individual.update_state(counter)):
				counter += 1

			if(counter < max_cycle):
				#stable, not chaotic or cyclic
				ham = hamming(individual.nodes[counter-1],target_attractors[set_index])
				#print individual.nodes[counter-1]
				this_fitness = (1-(ham/float(len(target_attractors[set_index])))) #raise to the 5th
				fitness_values.append(this_fitness)
			else:
				fitness_values.append(0) #zero fitness for chaotic/cyclic state

	#print fitness_values
	tot_fitness = sum(fitness_values)
	# tot_starting_attractors = 0
	# for attractor_set in range(num_attractor_sets):
	# 	for attractor in attractor_sets[attractor_set]:
	# 		tot_starting_attractors+=1
	tot_starting_attractors = 11*num_attractor_sets

	return tot_fitness/tot_starting_attractors

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

	#print "\nPart one complete!"
	#if(best.fitness>-1):
	#	print "The network that produced the most accurate attractors had fitness: " , best.fitness
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

	#pickle.dump( best, open( "best_network.pickle", "wb" ) )
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

def average_modularity(population):
	if(len(population))==0:
		return 0
	total=0
	for individual in population:
		total+=individual.measure_modularity()
	return total/len(population)
def average_connectivity(population):
	total=0
	if(len(population))==0:
		return 0
	for individual in population:
		total+=individual.get_connectedness()
	return float(total)/len(population)
def average_fitness(population):
	if(len(population))==0:
		return 0
	total=0
	for individual in population:
		total+=individual.fitness
	return float(total)/len(population)
#@profile
def det_pareto(max_cycle, pop_size, generations,mu,p,run_number,num_runs,num_targets,population,number_perfect_networks,attractor_sets):
	'''
	I will evolve populations of GRNs mirroring the initial results
	"Specialization Increases Modularity" in the original paper, but using an
	Age-Fitenss Pareto Optimization algorithm, and deterministic start attractors
	Returns the Q value of the first network found with fitness 1
	'''
	# plt.ion()
	# plt.show()
	targetA = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	#initial network: 200 networks with identical randomized edges:
	network_size = len(targetA)
	#initial_edges = model.GRN.initialize_edges(network_size,network_size)

	#Find fitness for each individual:
	for individual in population:
		individual.fitness = evaluate_network(individual,max_cycle,num_targets,attractor_sets)

	#population[0].visualize_network(targetA,targetA,20)
	#evolutionary loop is initiated:
	best = population[0]
	#print best.fitness
	gens=0
	best_networks= list()
	#while(len(best_networks)<number_perfect_networks):
	for gen in range(generations):
		if(gens%100==0):
			print "targets: "+str(number_perfect_networks)+" just passed "+str(gens)+" generations"
			print "best fitness so far: "+str(best.fitness)
		gens+=1
		#each network is evaluated, and mutated
		next_gen = []
		#for individual in population:
		rand.shuffle(population)
		for i in range(len(population)):
			individual=population[i]
			individual.genetic_age+=1
			# #crossover
			if(i<len(population)/2):# and num_targets>1):
				if(i%2==0):
					model.GRN.crossover(individual,population[i+1],5) #make sure this is the correct index
					#print 'genage:'+str(individual.genetic_age)
					#print 'genage:'+str(population[i+1].genetic_age)
					new_age= max(individual.genetic_age,population[i+1].genetic_age)
					#print 'new_age:'+str(new_age)
					individual.genetic_age=new_age
					population[i+1].genetic_age=new_age
					if(evaluate_network(individual,max_cycle,num_targets,attractor_sets)>best.fitness):
						best=individual
					if(evaluate_network(population[i+1],max_cycle,num_targets,attractor_sets)>best.fitness):
						best=population[i+1]
			else:
				child = individual.copy()
				child.perturb(mu)
				child.fitness = evaluate_network(child,max_cycle,num_targets,attractor_sets)
				if child.fitness > best.fitness:
					best = child
				#print best.fitness
				next_gen.append(child)
		population.extend(next_gen)
		
		#one extra random network is added at zero age:
		new_individual = model.GRN(targetA,max_cycle,model.GRN.initialize_edges(network_size,network_size))
		new_individual.fitness = evaluate_network(new_individual,max_cycle,num_targets,attractor_sets)
		population.append(new_individual)
		if new_individual.fitness > best.fitness:
			best = new_individual
			#print "random individual had best: ",best.fitness
			#print "new best with fitness: ",best.fitness

		#check for termination:
		if(best.fitness==1):
			print "new best"
			best.visualize_network
			#print "optimal fitness found"
			#print "network with fitness 1 found"
			best_networks.append(best)
			#remove other networks of same age
			population.remove(best)
			for ind in population:
				if ind.genetic_age==best.genetic_age:
					population.remove(ind)
			best = population[0]

		#now our population is of size 2k+1, time for tournaments:
		eliminated = []
		while(len(population)>pop_size):
			individualA = rand.choice(population)
			individualB = rand.choice(population)
			#total_tournament(population,eliminated)
			tournament(population,{individualA,individualB},eliminated)
			
			if(entirely_non_dominated(population)):
				pop_size=len(population)
				print "entirely_non_dominated"
 		#pareto_visualization(population,eliminated)
		#update_progress((gen*1.0*run_number)/((generations-1)*num_runs))
		#update_progress((run_number*1.0)/(num_runs))
		#print " Population size: ",len(population)
	if(len(best_networks)>0):
		return population,best_networks
	else:
		return population,[best]

def main():
	target_attractors=[[-1,1,-1,1,-1,1,-1,1,-1,1],
					   [-1,1,-1,1,-1,-1,1,-1,1,-1]]

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
	# attractor_sets=[list(),list()]
	# attractor_sets[0].append(target_attractors[0])
	# attractor_sets[1].append(target_attractors[1])
	# print attractor_sets
	# for i in range(10):
	# 	new_attractor = np.random.randint(2,size=10)
	# 	new_attractor[new_attractor == 0] = -1
	# 	new_attractor=new_attractor.tolist()
	# 	attractor_sets[0].append(new_attractor)
	# for i in range(10):
	# 	new_attractor = np.random.randint(2,size=10)
	# 	new_attractor[new_attractor == 0] = -1
	# 	new_attractor=new_attractor.tolist()
	# 	while ((new_attractor in attractor_sets[0]) or (new_attractor in attractor_sets[1])):
	# 		new_attractor = np.random.randint(2,size=10)
	# 		new_attractor[new_attractor == 0] = -1
	# 		new_attractor=new_attractor.tolist()
	# 	attractor_sets[1].append(new_attractor)
	# print attractor_sets

	seedsfile=sys.argv[1]
	outfile1=sys.argv[2]
	outfile2=sys.argv[3]
	number_perfect_networks=int(sys.argv[4])

	pop_size =100 #target number of nondominated individuals
	generations = 500
	mu = 0.05
	p=0.15

	#rand.seed("this is a seed") #for safety-harness
	#q_values_single = []
	#q_values_two = []
	#fitness_single = []
	#fitness_two = []
	trial_counter=0
	final_population_single=[]
	final_population_two=[]
	max_cycle = 50 #just let me test this
	
	with open(seedsfile+'.pickle', 'rb') as handle:
		seeds = pickle.load(handle)
	for seed in seeds:
		trial_counter+=1
		rand.seed(seed)
		new_network = model.GRN(np.array([-1,1,-1,1,-1,1,-1,1,-1,1]),max_cycle)
		print new_network.measure_modularity()
		#initialize population
		population=list()
		for i in range(int(pop_size)):
			new_network = model.GRN(np.array([-1,1,-1,1,-1,1,-1,1,-1,1]),max_cycle)
			population.append(new_network) #initially randomized

		#trial for target A only
		generations=500
		population,best_networks = det_pareto(max_cycle, pop_size, generations,mu,p,trial_counter,len(seeds),1,population,number_perfect_networks,attractor_sets)
		#q_values_single.append(average_modularity(best_networks))
		#fitness_single.append(average_fitness(best_networks))

		print "[targets: "+sys.argv[4]+"] for target A only, modularity: ",str(average_modularity(best_networks))," connections: "+str(average_connectivity(best_networks))
		population.extend(best_networks)
		final_population_single.append(population)

		#for i in range(len(best_networks)):
		pickle.dump(final_population_single,open('networks/populationsA.pickle','wb'))
		#and trial for target A and B
		generations=1500
		population.extend(best_networks)
		#rand.seed(seed)
		population,best_networks = det_pareto(max_cycle, pop_size, generations,mu,p,trial_counter,len(seeds),2,population,number_perfect_networks,attractor_sets)
		#q_values_two.append(average_modularity(best_networks))
		#fitness_two.append(average_fitness(best_networks))
		print "[targets: "+sys.argv[4]+"] for target A and B, modularity: ",str(average_modularity(best_networks))," connections: "+str(average_connectivity(best_networks))		
		population.extend(best_networks)
		final_population_two.append(population)
		#for i in range(len(best_networks)):
		pickle.dump(final_population_two,open('networks/populationsB.pickle','wb'))
		
		# pickle.dump( fitness_single, open( 'output/fitness_single.pickle', "wb" ) )
		# pickle.dump( fitness_two, open( 'output/fitness_two.pickle', "wb" ) )	
		# pickle.dump( q_values_single, open( 'output/'+outfile1+".pickle", "wb" ) )
		# pickle.dump( q_values_two, open( 'output/'+outfile2+".pickle", "wb" ) )
		print "finished trial ",trial_counter
main()