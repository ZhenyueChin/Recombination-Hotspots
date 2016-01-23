#This file tests a network. Pass in initial configurations, see the results.
#mostly a sanity check
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import model
import sys
import pickle
import model
from scipy import stats
max_cycle=20
attractor_sets = [[ 	[-1,1,-1,1,-1,1,-1,1,-1,1],
						[1,1,-1,1,-1,1,-1,1,-1,1],
						[-1,-1,-1,1,-1,1,-1,1,-1,1],
						[-1,1,1,1,-1,1,-1,1,-1,1],
						[-1,1,-1,-1,-1,1,-1,1,-1,1],
						[-1,1,-1,1,1,1,-1,1,-1,1],
						[-1,1,-1,1,-1,-1,-1,1,-1,1],
						[-1,1,-1,1,-1,1,1,1,-1,1],
						[-1,1,-1,1,-1,1,-1,-1,-1,1],
						[-1,1,-1,1,-1,1,-1,1,1,1],
						[-1,1,-1,1,-1,1,-1,1,-1,-1],
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

def get_top_performers(cutoff,population,phase):
	'''
	phase can be 1 (part A) or 2 (part B)
	'''
	fitnesses = [model.GRN.evaluate_network(individual, max_cycle, phase ,attractor_sets) for individual in population]
	percentile95=np.percentile(fitnesses, cutoff)
	top_performers = [individual for individual in population if individual.fitness>=percentile95]

	# top_fitnesses = [ind.fitness for ind in top_performers]
	# # fit = stats.norm.pdf(top_fitnesses, np.mean(top_fitnesses), np.std(top_fitnesses))  #this is a fitting indeed

	# # plt.plot(sorted(top_fitnesses),fit,'-o')

	# plt.hist(top_fitnesses,normed=True)      #use this to draw histogram of your data

	# plt.show() 

	return top_performers

def print_summary_stats(top_performers):

	print "number of top performers: "+str(len(top_performers))
	print "avg fitness for percentile: "+str(np.mean([i.fitness for i in top_performers]))
	print "avg mod for percentile: "+str(np.mean([i.measure_modularity() for i in top_performers]))
def compare_two_sets(exp1,exp2):
	# v = v[1:]
	# for el in range(10):
	# 	v[el] = int(v[el])

	# nodes = np.array(v)
	# max_cycle=20
	
	# with open('best_network.pickle', 'rb') as handle:
 #  		individual = pickle.load(handle)

	# print "the network:" 
	# print individual.edges

	
 #  	individual.nodes=nodes
 #  	print individual.nodes
 #  	individual.visualize_network()
	# counter = 0
	# while(counter <= max_cycle and individual.update_state()):
	# 	counter += 1
	# 	print individual.nodes
	# 	individual.visualize_network()

	# if(counter <= max_cycle):
	# 	print "stable"
	# 	print individual.nodes
	# else:
	# 	print "chaotic or cyclic"

	# individual.visualize_network()


	percentile = 95
	trial_1_populationsA = []
	trial_1_populationsB = []
	trial_2_populationsA = []
	trial_2_populationsB = []
	
	with open('networks/'+exp1+'/run2/populationsA.pickle', 'rb') as handle:
  		for pop in pickle.load(handle):
  			trial_1_populationsA+=get_top_performers(percentile,pop,1)
  		print len(trial_1_populationsA)

  	plt.hist([i.fitness for i in trial_1_populationsA],normed=True)      #use this to draw histogram of your data

	plt.show() 


  	with open('networks/populationsA1.pickle', 'rb') as handle:
  		for pop in pickle.load(handle):
  			trial_1_populationsB+=get_top_performers(percentile,pop,1)

  	with open('networks/populationsB1.pickle', 'rb') as handle:
  		for pop in pickle.load(handle):
  			trial_1_populationsB+=get_top_performers(percentile,pop,2)
  	# with open('networks/'+exp1+'/populationsB.pickle', 'rb') as handle:
  	# 	trial_1_populationsB+=(pickle.load(handle))[0]

  	with open('networks/populationsA2.pickle', 'rb') as handle:
  		for pop in pickle.load(handle):
  			trial_2_populationsA+=get_top_performers(percentile,pop,1)
  	# with open('networks/'+exp2+'/populationsB.pickle', 'rb') as handle:
  	# 	trial_2_populationsA+=(pickle.load(handle))[0]

  	with open('networks/populationsB2.pickle', 'rb') as handle:
  		for pop in pickle.load(handle):
  			trial_2_populationsB+=get_top_performers(percentile,pop,2)
  	# with open('networks/'+exp2+'/populationsB.pickle', 'rb') as handle:
  	# 	trial_2_populationsB+=(pickle.load(handle))[0]

  	print len(trial_1_populationsA)
  	print len(trial_1_populationsB)
  	print len(trial_2_populationsA)
  	print len(trial_2_populationsB)

  	print("\n"+exp1+"A:")
  	print_summary_stats(trial_1_populationsA)
  	print("\n"+exp1+"B:")
  	print_summary_stats(trial_1_populationsB)
  	print("\n"+exp2+"A:")
  	print_summary_stats(trial_2_populationsA)
  	print("\n"+exp2+"B:")
  	print_summary_stats(trial_2_populationsB)
	
	#trim population down to unique geneological trees? difficult when crossing over (multiple networks with same age)
	# print "\n"+exp1+" part A"
	# trial_1_top_performersA = get_top_performers(percentile,trial_1_populationsA,1)
	# print "\n"+exp1+" part B"
	# trial_1_top_performersB = get_top_performers(percentile,trial_1_populationsB,2)

	# print "\n"+exp2+" part A"
	# trial_2_top_performersA = get_top_performers(percentile,trial_2_populationsA,1)
	# print "\n"+exp2+" part B"
	# trial_2_top_performersB = get_top_performers(percentile,trial_2_populationsB,2)
	trial_1_top_performersA = trial_1_populationsA
	trial_1_top_performersB = trial_1_populationsB
	trial_2_top_performersA = trial_2_populationsA
	trial_2_top_performersB = trial_2_populationsB

	#t-test
	print "\nt value for comparing modularity, part A"
	t, p = stats.ttest_ind([i.measure_modularity() for i in trial_1_top_performersA],[i.measure_modularity() for i in trial_2_top_performersA])
	print "ttest_ind: t = %g  p = %g" % (t, p)
	print "t value for comparing modularity, part B"
	t, p = stats.ttest_ind([i.measure_modularity() for i in trial_1_top_performersB],[i.measure_modularity() for i in trial_2_top_performersB])
	print "ttest_ind: t = %g  p = %g" % (t, p)

	print "t value for comparing fitness, part A"
	t, p = stats.ttest_ind([i.fitness for i in trial_1_top_performersA],[i.fitness for i in trial_2_top_performersA])
	print "ttest_ind: t = %g  p = %g" % (t, p)
	print "t value for comparing fitness, part B"
	t, p = stats.ttest_ind([i.fitness for i in trial_1_top_performersB],[i.fitness for i in trial_2_top_performersB])
	print "ttest_ind: t = %g  p = %g" % (t, p)

	print "\nt value for comparing modularity between parts A and B in "+exp1
	t, p = stats.ttest_ind([i.measure_modularity() for i in trial_1_top_performersA],[i.measure_modularity() for i in trial_1_top_performersB])
	print "ttest_ind: t = %g  p = %g" % (t, p)

	print "\nt value for comparing modularity between parts A and B in "+exp2
	t, p = stats.ttest_ind([i.measure_modularity() for i in trial_2_top_performersA],[i.measure_modularity() for i in trial_2_top_performersB])
	print "ttest_ind: t = %g  p = %g" % (t, p)


  	individual=trial_1_top_performersA[0]
	target_stateA = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
	individual.rectangle_visualization(attractor_sets[0],target_stateA, "TargetA")

	for start_state in attractor_sets[0]:
  		individual.visualize_network(start_state,target_state,max_cycle)
  		temp = raw_input("enter to end")
def list_values(filename):
	# compare_two_sets("E2","E3")
	with open(filename, 'rb') as handle:
  		for pop in pickle.load(handle):
  			# print len(pop)
  			print pop
def count_values(filenames):
	'''
	pot -> partitions over time
	pop -> tuple [generation,index]
	'''
	cumulative=False
	records_per_file = 18751
	number_bins = 15
	pot=[[0]*9]*number_bins
	indexes=[0]*9
	counter=0
	for filename in filenames:
		with open(filename, 'rb') as handle:
			row=0
			last_pop=0
	  		for pop in pickle.load(handle):
	  			# print len(pop)
	  			indexes[pop[1]-1]+=1
	  			counter+=1

	  			if counter%(records_per_file/number_bins)==0:
	  				#pot.append(indexes)
	  				#print pot[row]
	  				#print indexes

	  				pot[row] = [x + y for x, y in zip(pot[row], indexes)]
	  				if( not cumulative):
	  					indexes=[0]*9
	  				row+=1
	  				
	  			last_pop = pop[0]
	  	print counter
	#pot[0][4]-=70
	tot=0
	for i in pot:
		print str(i)+str(sum(i))
		tot+=sum(i)
	print tot
  	return pot
def rectangle_visualization(pot):
	'''
	Shows the network behavior by timestep, with target
	'''
	fig, ax = plt.subplots()
	plt.title("Partition Location over time, E5",fontsize=20)
	
	
	plt.imshow(pot, cmap=plt.cm.gray, aspect='auto',interpolation='nearest')

	

	# We need to draw the canvas, otherwise the labels won't be positioned and 
	# won't have values yet.
	fig.canvas.draw()
	plt.ylabel('Generation')
	plt.xlabel('Partition Location')
	ax.set_xticklabels([i for i in range(-1,10,2)])
	ax.set_yticklabels([i for i in range(-50,800,100)])


	plt.show()
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

def main(v):
	#compare_two_sets("E2","E2")
	# partitions_over_time = count_values(['networks/crossovers4_1_0.pickle',
	# 									'networks/crossovers4_1_1.pickle',
	# 									'networks/crossovers4_1_2.pickle',
	# 									'networks/crossovers4_2_0.pickle',
	# 									'networks/crossovers4_2_1.pickle',
	# 									'networks/crossovers4_2_2.pickle',
	# 									'networks/crossovers4_3_0.pickle',
	# 									'networks/crossovers4_3_1.pickle',
	# 									'networks/crossovers4_3_2.pickle',
	# 									'networks/crossovers4_4_0.pickle',
	# 									'networks/crossovers4_4_1.pickle',
	# 									'networks/crossovers4_4_2.pickle',])
	partitions_over_time = count_values(['networks/E5/run4(allpairs)/crossovers5_1_0.pickle',
										'networks/E5/run4(allpairs)/crossovers5_1_1.pickle',
										'networks/E5/run4(allpairs)/crossovers5_1_2.pickle',
										'networks/E5/run4(allpairs)/crossovers5_3_0.pickle',
										'networks/E5/run4(allpairs)/crossovers5_3_1.pickle',
										'networks/E5/run4(allpairs)/crossovers5_3_2.pickle',
										'networks/E5/run4(allpairs)/crossovers5_2_0.pickle',
										'networks/E5/run4(allpairs)/crossovers5_2_1.pickle',
										'networks/E5/run4(allpairs)/crossovers5_2_2.pickle',
										'networks/E5/run4(allpairs)/crossovers5_4_0.pickle',
										'networks/E5/run4(allpairs)/crossovers5_4_1.pickle',
										'networks/E5/run4(allpairs)/crossovers5_4_2.pickle',
										])

	rectangle_visualization(partitions_over_time)

	# with open('networks/populationsA2.pickle', 'rb') as handle:
 #  		for pop in pickle.load(handle):
 #  			for i in range(len(pop)):
 #  				if(sum(pop[i].crossover_preference)>1.01 or sum(pop[i].crossover_preference)<0.99):
	#   				print '\n'
	#   				print sum(pop[i].crossover_preference)
	#   				print pop[i].crossover_preference
	#   			print pop[i].crossover_preference
	# trial1 = '5'
	# trial2 = '1'
	# bestsA=[]
	# for core in range(1,5):
	# 	for trial in range(0,3):
	# 		with open('networks/E'+trial1+'/run3(allpairs)/fitCurve'+trial1+'_2_'+str(core)+'_'+str(trial)+'.pickle', 'rb') as handle:
	# 	  		pop = pickle.load(handle).pop()
	# 	  		bestsA.append(pop.fitness)
	# 	  		if(pop.fitness==1):
	# 	  			print "got one"
	# 	  			toShow = pop

	# bestsB = []
	# for core in range(1,5):
	# 	for trial in range(0,3):
	# 		with open('networks/E'+trial2+'/run3(allpairs)/fitCurve'+trial2+'_2_'+str(core)+'_'+str(trial)+'.pickle', 'rb') as handle:
	# 	  		pop = pickle.load(handle).pop()
	# 	  		bestsB.append(pop)
	# print "avg fitness for "+trial1+": "+str(np.mean([i for i in bestsA]))
	# print "avg fitness for "+trial2+": "+str(np.mean([i for i in bestsB]))

	# print "\nt value for comparing fitness:"
	# t, p = stats.ttest_ind(bestsA,bestsB)
	# print "ttest_ind: t = %g  p = %g" % (t, p)


	# toShow.rectangle_visualization(generate_permutations([-1,1,-1,1,-1,-1,1,-1,1,-1]),[-1,1,-1,1,-1,-1,1,-1,1,-1], "TargetB, Fitness = "+str(toShow.fitness))
	# toShow.visualize_network([-1,-1,-1,1,-1,1,1,-1,1,-1],[-1,1,-1,1,-1,-1,1,-1,1,-1],20)
main(sys.argv)