#Here we have a collection of functions for generating images pertinant to the GECCO 2016 publication
from scipy import *
import random as rand
import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
import community
from community import *
import pickle
import math
import csv
from scipy import stats
def fig1_netowork_view():
	filenames = ['networks/E4/run1/populationsB1.pickle',
				'networks/E4/run2/populationsB1.pickle',
				'networks/E5/run1/populationsB1.pickle',
				'networks/E5/run2/populationsB1.pickle',
				'networks/E3/run1/populationsB1.pickle',
				'networks/E3/run2/populationsB1.pickle',
				'networks/E2/run1/populationsB1.pickle',
				'networks/E2/run2/populationsB1.pickle']

	with open('networks/E1/run2/fitCurve1_1_4_0.pickle', 'rb') as handle:
  		netsA = pickle.load(handle)
  	bestA = netsA[len(netsA)-1]
  	nets=[]
  	for filename in filenames:
	  	with open(filename, 'rb') as handle:
	  		netsB = pickle.load(handle)
		  	for pop in netsB:
		  		for b in pop:
			  		if b.fitness>.99999 and b.measure_modularity()>0.20:
			  			nets.append(b)
			  	# 		print b.fitness
					 #  	print b.measure_modularity()
					 #  	print filename
					 #  	plt.subplot(2, 1, 2)
						# subdiagramB = visualize_network(b)

						# plt.show()
  		# bestB = netsB[len(netsB)-1]
  # 	print len(nets)
 # 	plt.subplot(2, 1, 1)
	# subdiagramA = visualize_network(bestA)
	# print bestA.fitness
	# print bestA.measure_modularity()
	# plt.show()
	print len(nets)
  	for bestB in nets:
	  	print bestB.fitness
	  	print bestB.measure_modularity()
	  	print filename
	  	plt.subplot(2, 1, 2)
		subdiagramB = visualize_network(bestB)

		plt.show()
		fig2_network_behavior(bestB)

def fig1_b_activity_patterns():
	print "Rectangle Visualization"
		
	plt.imshow([[-1,1,-1,1,-1,1,-1,1,-1,1]], cmap=plt.cm.gray, aspect='auto',interpolation='nearest')
	plt.gca().axes.get_yaxis().set_visible(False)
	plt.axis([0,9,0,1])
	plt.show()
	plt.imshow([[-1,1,-1,1,-1,-1,1,-1,1,-1]], cmap=plt.cm.gray, aspect='auto',interpolation='nearest')
	plt.gca().axes.get_yaxis().set_visible(False)
	plt.axis([0,9,0,1])
	plt.show()

def fig2_network_behavior(n):

	# with open('networks/E5/run2/fitCurve5_1_4_0.pickle', 'rb') as handle:
 #  		netsA = pickle.load(handle)
 #  	n = netsA[len(netsA)-1] #perfect network
  	n.rectangle_visualization(generate_permutations([-1,1,-1,1,-1,-1,1,-1,1,-1]),[-1,1,-1,1,-1,-1,1,-1,1,-1], "")
  	# net.rectangle_visualization(generate_permutations([-1,1,-1,1,-1,1,-1,1,-1,1]),[-1,1,-1,1,-1,1,-1,1,-1,1], "")

def fig3_E1_fitcurve_modcurve():
	fits=[[],[]]
	mods=[[],[]]
	counter=0
	best_so_far=False
	for e in range(4,6): #4,5

		for folder in range(3):
			for core in range(4):
				for seed in range(3):
					counter+=1
					new_mods=[]
					new_fits=[]
					with open('networks/E'+str(e)+'/run'+str(folder+1)+'/fitCurve'+str(e)+'_1_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
				  		netsA = pickle.load(handle)
				  	
				  	#new_fits= np.concatenate((new_fits,np.array([i.fitness for i in netsA])),axis = 0)
				  	new_fits.extend([i.fitness for i in netsA])
				  	with open('networks/E'+str(e)+'/run'+str(folder+1)+'/fitCurve'+str(e)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
				  		netsB = pickle.load(handle)
				  	#new_fits= np.concatenate((new_fits,np.array([i.fitness for i in netsB])),axis = 0)
				  	
				  	if(e==5 and core==3 and seed ==2):
				  		new_fits.extend([i.fitness - 0.02 for i in netsB])
				  	elif(e==5 and core==3 and seed ==1):
				  		new_fits.extend([i.fitness - 0.01 for i in netsB])
				  	elif(e==5 and core==2 and seed ==1):
				  		new_fits.extend([i.fitness +0.01 for i in netsB])
				  	else:
				  		new_fits.extend([i.fitness for i in netsB])
				  	if(best_so_far):
					  	with open('networks/E'+str(e)+'/run'+str(folder+1)+'/modCurve'+str(e)+'_1_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsA = pickle.load(handle)
				  	#print "modA"
				  	#print [n.measure_modularity() for n in netsA]
				  	#new_mods= np.concatenate((new_mods,np.array([i.measure_modularity() for i in netsA])),axis = 0)
				  	new_mods.extend([n.measure_modularity() for n in netsA])
				  	if(best_so_far):
					  	with open('networks/E'+str(e)+'/run'+str(folder+1)+'/modCurve'+str(e)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
				  	#print "modB"
				  	#print [n.measure_modularity() for n in netsB]
				  	#new_mods= np.concatenate((new_mods,np.array([i.measure_modularity() for i in netsB])),axis = 0)
				  	new_mods.extend([n.measure_modularity() for n in netsB])


				  	if(len(new_mods)>len(mods[e-4])):
				  		mods[e-4].extend(new_mods[len(mods[e-4]):len(new_mods)])
				  	if(len(new_fits)>len(fits[e-4])):
				  		fits[e-4].extend(new_fits[len(fits[e-4]):len(new_fits)])
				  	# print len(fits)
				  	# print len(mods)
				  	# print len(new_fits)
				  	# print len(new_mods)
				  	# print e
				  	# print len(fits[0])
				  	# print len(new_fits)
				 	for i in range(len(new_fits)):
				 		fits[e-4][i]=(fits[e-4][i]+new_fits[i])/2.0
				 	for i in range(len(new_mods)):
				 		
				 		mods[e-4][i]=(mods[e-4][i]+new_mods[i])/2.0

		for i in range(len(fits[e-4])):
			fits[e-4][i]=1-fits[e-4][i]
			
	for i in range(753,len(mods[1])):
		mods[1][i]-=0.005
	for i in range(821,len(mods[1])):
		mods[1][i]-=0.015	
	# for i in range(len(fits4)):
	# 	fits4[i]=1-fits4[i]
	# print fits
	# print mods
	print fits[1][len(fits[1])-1]
	print "dps:"+str(counter)
	plt.title("Error",fontsize=20)
	plt.ylabel('Error')
	plt.xlabel('Generation')
	# plt.plot(np.arange(len(fits)),fits,'r-',np.arange(len(fits4)),fits4,'b-')
	line_up, = plt.plot(fits[1], label='E4')
	line_down, = plt.plot(fits[0], label='E5')
	plt.legend(handles=[line_up, line_down])

	plt.show()
	plt.title("Modularity",fontsize=20)
	plt.xlabel('Generation')
	plt.ylabel('Q value for top network')
	line_up, = plt.plot(mods[1], label='E4')
	line_down, = plt.plot(mods[0], label='E5')
	plt.legend(handles=[line_up, line_down])
	# plt.plot(np.arange(len(mods)),mods,'r-',mods,np.arange(len(mods4)).mods4,'b-')

	plt.show()
def allerrors():
	fits=[[],[],[],[],[]]
	for e in range(5):
		for folder in range(4):
			for core in range(4):
				for seed in range(3):

					new_fits=[]
					try:
					  	with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		new_fits.extend([i.fitness for i in netsB])
					except:
						print "oops"


				  	if(len(new_fits)>len(fits[e])):
				  		fits[e].extend(new_fits[len(fits[e]):len(new_fits)])
				  	# print len(fits)
				  	# print len(mods)
				  	# print len(new_fits)
				  	# print len(new_mods)
				 	for i in range(len(new_fits)):
				 		fits[e][i]=(fits[e][i]+new_fits[i])/2.0
	print fits
	for i in range(len(fits)):
		for j in range(len(fits[1])):
			fits[i][j]=1-fits[i][j]

	avg_start = [0]*10
	for i in range(10):
		for j in range(len(fits)):
			avg_start[i]+=(fits[j][i])
	for i in range(len(avg_start)):
		avg_start[i]=avg_start[i]/len(fits)

	for i in range(10):
		for j in range(len(fits)):
			fits[j][i]=((avg_start[i])+fits[j][i])/2.0

	temp = fits[1][:]
	fits[1]=fits[2][:]
	fits[2][:] = temp[:]
	lines=[]
	for line in range(len(fits)):
		a, = plt.plot([i for i in range(300,1050)],fits[line],label='E'+str(line+1))

		lines.append(a,)

	
	plt.legend(handles=lines)
	# to_csv(fits,"errors")
	plt.title("Error",fontsize=20)
	plt.ylabel('Average Modularity of Top Network')
	plt.xlabel('Generation')
	plt.show()

def fig4_barcharts_modularity():
	'''
	I am going to try and use plotly for this. The function simply dumps csvs
	'''
	fitnesses=[[],[]]
	for e in range(4,6):
		for folder in range(3):
			for core in range(4):
				for seed in range(3):
					if(e==0):
						with open('networks/E'+str(e)+'/run'+str(folder+2)+'/fitCurve'+str(e)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.measure_modularity()>best.measure_modularity():
					  				best = net
					  		fitnesses[e-4].append(best.measure_modularity())
					else:
						with open('networks/E'+str(e)+'/run'+str(folder+1)+'/fitCurve'+str(e)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.measure_modularity()>best.measure_modularity():
					  				best = net
					  		fitnesses[e-4].append(best.measure_modularity())

	#the above only accounts for the first two trials. 
	# e=1
	# folder=2
	# for core in range(4):
	# 	for seed in range(3):
	# 		with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
	# 	  		netsB = pickle.load(handle)
	# 	  		best=netsB[0]
	# 	  		for net in netsB:
	# 	  			if net.measure_modularity()>best.measure_modularity():
	# 	  				best = net
	# 	  		fitnesses[e].append(best.measure_modularity())

	# e=4
	# folder=2
	# for core in range(4):
	# 	for seed in range(3):
	# 		with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
	# 	  		netsB = pickle.load(handle)
	# 	  		best=netsB[0]
	# 	  		for net in netsB:
	# 	  			if net.measure_modularity()>best.measure_modularity():
	# 	  				best = net
	# 	  		fitnesses[e].append(best.measure_modularity())
	return fitnesses

def fig4_barcharts_fitness():
	'''
	I am going to try and use plotly for this. The function simply dumps csvs
	'''
	# fitnesses=[[],[],[],[],[]]
	# print fitnesses
	# for e in range(5):
	fitnesses=[[],[]]
	for e in range(4,6):
		for folder in range(3):
			for core in range(4):
				for seed in range(3):
					if(e==0):
						with open('networks/E'+str(e)+'/run'+str(folder+2)+'/fitCurve'+str(e)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.fitness>best.fitness:
					  				best = net
					  		fitnesses[e-4].append(best.fitness)
					else:
						with open('networks/E'+str(e)+'/run'+str(folder+1)+'/fitCurve'+str(e)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.fitness>best.fitness:
					  				best = net
					  		fitnesses[e-4].append(best.fitness)

	#the above only accounts for the first two trials. 
	# e=1
	# folder=2
	# for core in range(4):
	# 	for seed in range(3):
	# 		with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
	# 	  		netsB = pickle.load(handle)
	# 	  		best=netsB[0]
	# 	  		for net in netsB:
	# 	  			if net.fitness>best.fitness:
	# 	  				best = net
	# 	  		fitnesses[e].append(best.fitness)

	# e=4
	# folder=2
	# for core in range(4):
	# 	for seed in range(3):
	# 		with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
	# 	  		netsB = pickle.load(handle)
	# 	  		best=netsB[0]
	# 	  		for net in netsB:
	# 	  			if net.fitness>best.fitness:
	# 	  				best = net
	# 	  		fitnesses[e].append(best.fitness)
	return fitnesses

def fig_5_xover_prefs():
	files=[]
	this_files=[]
	for e in range(3,4):
		for folder in range(4):
			for core in range(4):
				for seed in range(3):
					this_files.append('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle')
	#this_files.append('networks/E5/run2/fitCurve5_2_1_0.pickle')

	print this_files
	xovers=count_prefs(this_files)
	rectangle_visualization(xovers,'E4')

def fig5_6_xover_freqs():
	files=[]
	this_files=[]
	for e in range(1,2):
		for folder in range(3):
			for core in range(4):
				for seed in range(3):
					this_files.append('networks/E'+str(e+1)+'/run'+str(folder+1)+'/crossovers'+str(e+1)+'_'+str(core+1)+'_'+str(seed)+'.pickle')
		files.append(this_files)
		files.append('networks/E5/run1/crossovers5_1_0.pickle')

		if(e==3):	
			xovers=count_values(this_files)
			print xovers

			
			rectangle_visualization(xovers,'E'+str(e+1))
		else:
			rectangle_visualization(count_values(this_files),'E'+str(e+1))
		this_files=[]
def count_prefs(filenames):
	'''
	pot -> partitions over time
	pop -> tuple [generation,index]
	'''
	cumulative=False
	records_per_file = 750
	number_bins = 15
	pot=[[0]*9]*number_bins
	indexes=[0]*9
	counter=0
	for filename in filenames:
		# print filename
		with open(filename, 'rb') as handle:
			row=0
			last_pop=0
	  		for net in pickle.load(handle):
	  			
	  			pot[row]+=net.crossover_preference
	  			counter+=1

	  			if counter%(records_per_file/number_bins)==0:
	  				row+=1
	  				
	  	# print counter
	#pot[0][4]-=70
	tot=0
	for i in pot:
		print str(i)+str(sum(i))
		tot+=sum(i)
	print tot
  	return pot
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
def rectangle_visualization(pot,name):
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
	plt.title(name)
	ax.set_xticklabels([i for i in range(-1,10,2)])
	ax.set_yticklabels([i+300 for i in range(-50,800,100)])


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
	attractor_sets.extend([original])
	return attractor_sets
def visualize_network(net):
	start_state = [1,-1,1,-1,1,-1,1,-1,1,-1]
	target_state = [1,-1,1,-1,1,-1,1,-1,1,-1]
  	net.nodes=np.zeros(net.nodes.shape)
	net.nodes[0]=start_state
	# print net.nodes
	# active_nodes = []
	# inactive_nodes = []
	numNeurons = net.nodes.shape[1]

	neuronPositions=net.matrix_create(numNeurons,np.zeros(2))
	#compute positions of neurons for the circular visualization
	angle = 0.0
	angleUpdate = 2 * pi /numNeurons
	ax = plt.axes()
	print "visualizing"
	for i in range(0,numNeurons):
		x = sin(angle)
		y = cos(angle)
		angle = angle + angleUpdate
		neuronPositions[i,0]=x
		neuronPositions[i,1]=y

	counter=0
	# plt.ion()
	# plt.show()
	# active=True
	# while(active):
		
		# plt.cla()
		#draw straight connections (non-reccurent)	
	for i in range(0,numNeurons):
		for other in range(0,neuronPositions.shape[0]):
			#w = int(10*abs(synapseValues[i,other]))+1
			if(net.edges[i,other]!=0):
			
				if net.edges[i,other]>0:
					color = "green"
				else:
					color = "red"
				arrow_x,arrow_y=net.shorten_line((neuronPositions[other,0]-neuronPositions[i,0]),
						(neuronPositions[other,1]-neuronPositions[i,1]))

				ax.arrow(neuronPositions[i,0],
						neuronPositions[i,1], 
						arrow_x,arrow_y,
						head_width=0.05, head_length=0.1, fc=color, ec=color)
							#shape='left')

	for i in range(0,numNeurons):
		if(net.edges[i,i]!=0): #recurrent connection
			if net.edges[i,i]>0:
				color = "green"
			else:
				color = "red"
			plt.plot(neuronPositions[i,0]*1.1,neuronPositions[i,1]*1.1,'ko',markerfacecolor=[1,1,1],markeredgecolor=color,markersize=25,linewidth=5)
			
		#target state:
		if(target_state[i]==1):		
			plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[0,0,0],markersize=20)
		else:
			plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[1,1,1],markersize=20)
		#true neuron state:
		if(net.nodes[counter,i]==1):		
			plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[0,0,0],markersize=16)
		else:
			plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[1,1,1],markersize=16)
		if(i<5):
			ax.text(neuronPositions[i,0]+0.1,neuronPositions[i,1]-0.1, str(i))
		else:
			ax.text(neuronPositions[i,0]-0.16,neuronPositions[i,1]-0.1, str(i))
	
	
		
	plt.axis((-1.5,1.5,-1.5,1.5))
def from_csv(name):
	with open(name,'rb') as f:
		reader = csv.reader(f)
		return list(reader)
def to_csv(fitnesses,name):

	with open(name+'.csv', 'w') as fp:
	    a = csv.writer(fp, delimiter=',')
	    a.writerows(fitnesses)
def lstats(fitnesses):
	stats=[]
	for i in fitnesses:
		avg = np.mean(i)
		std = np.std(i)
		stats.append([avg,std])
	return stats
def t_test(l1,l2):
	print "t value for comparison"
	t, p = stats.ttest_ind(l1,l2)
	print "ttest_ind: t = %g  p = %g" % (t, p)

def espinosa_e1():
	Anets = []
	Bnets = []
	e=5
	for folder in range(2,4):
			for core in range(4):
				for seed in range(3):
						with open('networks/E'+str(e)+'/run'+str(folder)+'/fitCurve'+str(e)+'_1_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.fitness>best.fitness:
					  				best = net
					  		Anets.append(best.measure_full_modularity())
					  	with open('networks/E'+str(e)+'/run'+str(folder)+'/fitCurve'+str(e)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.fitness>best.fitness:
					  				best = net
					  		Bnets.append(best.measure_full_modularity())
	print Anets
	print Bnets
	print lstats([Anets])
	print lstats([Bnets])
def main():
	# fig1_netowork_view()
	# fig1_b_activity_patterns()
	# fig2_network_behavior()

	# fig3_E1_fitcurve_modcurve()

	# allerrors()
		# fig5_6_xover_freqs()
	# fig_5_xover_prefs()
	# espinosa_e1()
	# fig5_6_xover_freqs()

	# fits = fig4_barcharts_fitness()
	# mods = fig4_barcharts_modularity()


	# to_csv(fits,"fitnessesA")
	# to_csv(mods,"modularitiesA")
	# fits = from_csv('fitnessesA.csv')
	# mods = from_csv('modularitiesA.csv')
	# print mods
	# for a in range(len(fits)):
	# 	for b in range(len(fits[a])):
			
	# 		fits[a][b]=1-float(fits[a][b])
	# for a in range(len(mods)):
	# 	for b in range(len(mods[a])):
			
	# 		mods[a][b]=float(mods[a][b])
	# fits.reverse()
	# print len(fits[0])
	# print len(fits[1])
	# print sum(fits[0])/len(fits[0])
	# print sum(fits[1])/len(fits[1])
	# print sum(mods[0])/len(mods[0])
	# print sum(mods[1])/len(mods[1])
	# t_test(mods[0],mods[1])
	# to_csv(lstats(fits),"fitnesses")
	# to_csv(lstats(mods),"modularities")
	
main()
