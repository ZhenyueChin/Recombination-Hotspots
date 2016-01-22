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
from time import gmtime,strftime


def generate_permutations(original):
	'''
	generates all pair permutations of the original matrix
	'''
	attractor_sets=[]
	for i in range(10):
		new = original[:]
		new[i]=new[i]*(-1)

		attractor_sets.append(new)
		for j in range(i+1,10):
			newer = new[:]
			newer[j]=new[j]*(-1)
			attractor_sets.append(newer)
	return attractor_sets

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
def det_pareto(max_cycle, pop_size, generations,mu,p,run_number,num_runs,num_targets,population,number_perfect_networks,attractor_sets,core,E):
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
		model.GRN.evaluate_network(individual,max_cycle,num_targets,attractor_sets)

	#population[0].visualize_network(targetA,targetA,20)
	#evolutionary loop is initiated:
	best = population[0]
	most_modular=population[0]
	#print best.fitness
	gens=0
	best_networks= list()
	#while(len(best_networks)<number_perfect_networks):
	fit_curve=[]
	mod_curve=[]
	most_modular=population[0]
	for gen in range(generations):
		if(gens%100==0):
			print "targets: "+str(number_perfect_networks)+" just passed "+str(gens)+" generations: "+strftime("%Y-%m-%d %I:%M:%S")
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
			if(i<len(population)/2 and num_targets>1 and E!=1):
				if(i%2==0):
					xover_children = model.GRN.crossover(individual,population[i+1],E,core,run_number,gen) #make sure this is the correct index
					next_gen.extend(xover_children)
			else:
				child = individual.copy()
				child.perturb(mu,E)
				#print best.fitness
				next_gen.append(child)

		#one extra random network is added at zero age:
		new_individual = model.GRN(targetA,max_cycle,model.GRN.initialize_edges(network_size,network_size))
		next_gen.append(new_individual)

		for i in next_gen:
			model.GRN.evaluate_network(i,max_cycle,num_targets,attractor_sets)
			if i.fitness > best.fitness:
				best = i
				if (i.fitness == 1.0):
					with open('networks/first_best'+str(E)+'_'+str(sys.argv[4]+'_'+str(run_number))+'.pickle', 'rb') as handle:
						best_list = pickle.load(handle)
						best_list.append([gen,i])
					with open('networks/first_best'+str(E)+'_'+str(sys.argv[4])+'_'+str(run_number)+'.pickle', 'wb') as handle:
						pickle.dump(best_list,handle)
			if i.measure_modularity() > most_modular.measure_modularity():
				most_modular=i
		population.extend(next_gen)
		fit_curve.append(best)
		mod_curve.append(most_modular)



		#check for termination:
		# if(best.fitness==1):
		# 	print "new best"
		# 	best.visualize_network
		# 	#print "optimal fitness found"
		# 	#print "network with fitness 1 found"
		# 	best_networks.append(best)
		# 	#remove other networks of same age
		# 	population.remove(best)
		# 	for ind in population:
		# 		if ind.genetic_age==best.genetic_age:
		# 			population.remove(ind)
		# 	best = population[0]

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
	#dump the fit curve
	fit_file = open('networks/fitCurve'+str(E)+'_'+str(num_targets)+'_'+core+'_'+str(run_number)+'.pickle','wb')
	pickle.dump(fit_curve,fit_file)
	fit_file.close()
	mod_file = open('networks/modCurve'+str(E)+'_'+str(num_targets)+'_'+core+'_'+str(run_number)+'.pickle','wb')
	pickle.dump(mod_curve,mod_file)
	mod_file.close()
	print "finished part, best fitness so far: "+str(best.fitness)
	if(len(best_networks)>0):
		return population,best_networks
	else:
		return population,[best]

def main():
	
	E=5
	target_attractors=[[-1,1,-1,1,-1,1,-1,1,-1,1],
					   [-1,1,-1,1,-1,-1,1,-1,1,-1]]

	# attractor_sets = [[ [-1,1,-1,1,-1,1,-1,1,-1,1],
	# 					[1,1,-1,1,-1,1,-1,1,-1,1],
	# 					[-1,-1,-1,1,-1,1,-1,1,-1,1],
	# 					[-1,1,1,1,-1,1,-1,1,-1,1],
	# 					[-1,1,-1,-1,-1,1,-1,1,-1,1],
	# 					[-1,1,-1,1,1,1,-1,1,-1,1],
	# 					[-1,1,-1,1,-1,-1,-1,1,-1,1],
	# 					[-1,1,-1,1,-1,1,1,1,-1,1],
	# 					[-1,1,-1,1,-1,1,-1,-1,-1,1],
	# 					[-1,1,-1,1,-1,1,-1,1,1,1],
	# 					[-1,1,-1,1,-1,1,-1,1,-1,-1]
	# 					],
	# 				  [[-1,1,-1,1,-1,-1,1,-1,1,-1],
	# 				  [1,1,-1,1,-1,-1,1,-1,1,-1],
	# 				  [-1,-1,-1,1,-1,-1,1,-1,1,-1],
	# 				  [-1,1,1,1,-1,-1,1,-1,1,-1],
	# 				  [-1,1,-1,-1,-1,-1,1,-1,1,-1],
	# 				  [-1,1,-1,1,1,-1,1,-1,1,-1],
	# 				  [-1,1,-1,1,-1,1,1,-1,1,-1],
	# 				  [-1,1,-1,1,-1,-1,-1,-1,1,-1],
	# 				  [-1,1,-1,1,-1,-1,1,1,1,-1],
	# 				  [-1,1,-1,1,-1,-1,1,-1,-1,-1],
	# 				  [-1,1,-1,1,-1,-1,1,-1,1,1]
	# 				  ]]
	attractor_sets=[generate_permutations(target_attractors[0]),
					generate_permutations(target_attractors[1])
					]
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
	#generations = 500
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
	max_cycle = 20 #just let me test this
	
	
	with open(seedsfile+'.pickle', 'rb') as handle:
		seeds = pickle.load(handle)
	for seed in seeds:
		pickle.dump([],open('networks/first_best'+str(E)+'_'+str(sys.argv[4])+'_'+str(trial_counter)+'.pickle','wb'))
		pickle.dump([],open('networks/crossovers'+str(E)+'_'+str(sys.argv[4])+'_'+str(trial_counter)+'.pickle','wb'))
		
		rand.seed(seed)
		new_network = model.GRN(np.array([-1,1,-1,1,-1,1,-1,1,-1,1]),max_cycle)
		print new_network.measure_modularity()
		#initialize population
		population=list()
		for i in range(int(pop_size)):
			new_network = model.GRN(np.array([-1,1,-1,1,-1,1,-1,1,-1,1]),max_cycle)
			population.append(new_network) #initially randomized

		#trial for target A only
		generations=300
		#population,best_networks = det_pareto(max_cycle, pop_size, generations,mu,p,trial_counter,len(seeds),1,population,number_perfect_networks,attractor_sets,sys.argv[4],E)
		

		with open('networks/E1/run3(allpairs)/populationsA'+str(sys.argv[4])+'.pickle', 'rb') as handle:
			population = pickle.load(handle)[trial_counter]
		best_networks=[]
		print "[targets: "+sys.argv[4]+"] for target A only, modularity: ",str(average_modularity(best_networks))," connections: "+str(average_connectivity(best_networks))
		population.extend(best_networks)
		final_population_single.append(population)

		
		pickle.dump(final_population_single,open('networks/populationsA'+sys.argv[4]+'.pickle','wb'))
		#and trial for target A and B
		generations=750
		population.extend(best_networks)
		#rand.seed(seed)
		population,best_networks = det_pareto(max_cycle, pop_size, generations,mu,p,trial_counter,len(seeds),2,population,number_perfect_networks,attractor_sets,sys.argv[4],E)
		#q_values_two.append(average_modularity(best_networks))
		#fitness_two.append(average_fitness(best_networks))
		print "[targets: "+sys.argv[4]+"] for target A and B, modularity: ",str(average_modularity(best_networks))," connections: "+str(average_connectivity(best_networks))		
		population.extend(best_networks)
		final_population_two.append(population)
		#for i in range(len(best_networks)):
		pickle.dump(final_population_two,open('networks/populationsB'+sys.argv[4]+'.pickle','wb'))
		
		# pickle.dump( fitness_single, open( 'output/fitness_single.pickle', "wb" ) )
		# pickle.dump( fitness_two, open( 'output/fitness_two.pickle', "wb" ) )	
		# pickle.dump( q_values_single, open( 'output/'+outfile1+".pickle", "wb" ) )
		# pickle.dump( q_values_two, open( 'output/'+outfile2+".pickle", "wb" ) )
		print "finished trial ",trial_counter
		trial_counter+=1
main()