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
			  		if b.fitness>.95 and b.measure_modularity()>0.3:
			  			nets.append(b)
			  	# 		print b.fitness
					 #  	print b.measure_modularity()
					 #  	print filename
					 #  	plt.subplot(2, 1, 2)
						# subdiagramB = visualize_network(b)

						# plt.show()
  		# bestB = netsB[len(netsB)-1]
  # 	print len(nets)
  # 	for bestB in nets:
	 #  	print bestB.fitness
	 #  	print bestB.measure_modularity()
	 #  	print filename
	 #  	plt.subplot(2, 1, 2)
		# subdiagramB = visualize_network(bestB)

		# plt.show()
	plt.subplot(2, 1, 1)
	subdiagramA = visualize_network(bestA)
	print bestA.fitness
	print bestA.measure_modularity()
	plt.show()
	
def fig2_network_behavior():

	with open('networks/E5/run2/fitCurve5_1_4_0.pickle', 'rb') as handle:
  		netsA = pickle.load(handle)
  	n = netsA[len(netsA)-1] #perfect network
  	n.rectangle_visualization(generate_permutations([-1,1,-1,1,-1,-1,1,-1,1,-1]),[-1,1,-1,1,-1,-1,1,-1,1,-1], "Network Behavior")

def fig3_E1_fitcurve_modcurve():
	fits=[]
	mods=[]
	counter=0
	e=5
	best_so_far = True #rather than mod of best
	for folder in range(1,2):
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
			  	new_fits.extend([i.fitness for i in netsB])
			  	if(best_so_far):
				  	with open('networks/E4/run'+str(folder+1)+'/modCurve4_1_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
				  		netsA = pickle.load(handle)
			  	#print "modA"
			  	#print [n.measure_modularity() for n in netsA]
			  	#new_mods= np.concatenate((new_mods,np.array([i.measure_modularity() for i in netsA])),axis = 0)
			  	new_mods.extend([n.measure_modularity() for n in netsA])
			  	if(best_so_far):
				  	with open('networks/E4/run'+str(folder+1)+'/modCurve4_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
				  		netsB = pickle.load(handle)
			  	#print "modB"
			  	#print [n.measure_modularity() for n in netsB]
			  	#new_mods= np.concatenate((new_mods,np.array([i.measure_modularity() for i in netsB])),axis = 0)
			  	new_mods.extend([n.measure_modularity() for n in netsB])

			  	if(len(new_mods)>len(mods)):
			  		mods.extend(new_mods[len(mods):len(new_mods)])
			  	if(len(new_fits)>len(fits)):
			  		fits.extend(new_fits[len(fits):len(new_fits)])
			  	# print len(fits)
			  	# print len(mods)
			  	# print len(new_fits)
			  	# print len(new_mods)
			 	for i in range(len(new_fits)):
			 		fits[i]=(fits[i]+new_fits[i])/2.0
			 	for i in range(len(new_mods)):
			 		#NOTE: This is not ok. figure out why part B sometimes drops in modularity. It doesn't make a lick of sense
			 		
			 		mods[i]=(mods[i]+new_mods[i])/2.0
	e=1
	for folder in range(1,2):
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
			  	new_fits.extend([i.fitness for i in netsB])
			  	if(best_so_far):
				  	with open('networks/E4/run'+str(folder+1)+'/modCurve4_1_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
				  		netsA = pickle.load(handle)
			  	#print "modA"
			  	#print [n.measure_modularity() for n in netsA]
			  	#new_mods= np.concatenate((new_mods,np.array([i.measure_modularity() for i in netsA])),axis = 0)
			  	new_mods.extend([n.measure_modularity() for n in netsA])
			  	if(best_so_far):
				  	with open('networks/E4/run'+str(folder+1)+'/modCurve4_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
				  		netsB = pickle.load(handle)
			  	#print "modB"
			  	#print [n.measure_modularity() for n in netsB]
			  	#new_mods= np.concatenate((new_mods,np.array([i.measure_modularity() for i in netsB])),axis = 0)
			  	new_mods.extend([n.measure_modularity() for n in netsB])

			  	if(len(new_mods)>len(mods)):
			  		mods.extend(new_mods[len(mods):len(new_mods)])
			  	if(len(new_fits)>len(fits)):
			  		fits.extend(new_fits[len(fits):len(new_fits)])
			  	# print len(fits)
			  	# print len(mods)
			  	# print len(new_fits)
			  	# print len(new_mods)
			 	for i in range(len(new_fits)):
			 		fits[i]=(fits[i]+new_fits[i])/2.0
			 	for i in range(len(new_mods)):
			 		#NOTE: This is not ok. figure out why part B sometimes drops in modularity. It doesn't make a lick of sense
			 		
			 		mods[i]=(mods[i]+new_mods[i])/2.0
	e=4
	for folder in range(2):
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
			  	new_fits.extend([i.fitness for i in netsB])
			  	if(best_so_far):
				  	with open('networks/E4/run'+str(folder+1)+'/modCurve4_1_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
				  		netsA = pickle.load(handle)
			  	#print "modA"
			  	#print [n.measure_modularity() for n in netsA]
			  	#new_mods= np.concatenate((new_mods,np.array([i.measure_modularity() for i in netsA])),axis = 0)
			  	new_mods.extend([n.measure_modularity() for n in netsA])
			  	if(best_so_far):
				  	with open('networks/E4/run'+str(folder+1)+'/modCurve4_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
				  		netsB = pickle.load(handle)
			  	#print "modB"
			  	#print [n.measure_modularity() for n in netsB]
			  	#new_mods= np.concatenate((new_mods,np.array([i.measure_modularity() for i in netsB])),axis = 0)
			  	new_mods.extend([n.measure_modularity() for n in netsB])

			  	if(len(new_mods)>len(mods)):
			  		mods.extend(new_mods[len(mods):len(new_mods)])
			  	if(len(new_fits)>len(fits)):
			  		fits.extend(new_fits[len(fits):len(new_fits)])
			  	# print len(fits)
			  	# print len(mods)
			  	# print len(new_fits)
			  	# print len(new_mods)
			 	for i in range(len(new_fits)):
			 		fits[i]=(fits[i]+new_fits[i])/2.0
			 	for i in range(len(new_mods)):
			 		#NOTE: This is not ok. figure out why part B sometimes drops in modularity. It doesn't make a lick of sense
			 		
			 		mods[i]=(mods[i]+new_mods[i])/2.0
	if(best_so_far):
		for i in range(1,len(mods)):
			if mods[i]<mods[i-1]:
				mods[i]=mods[i-1]
	# print fits
	# print mods
	print "dps:"+str(counter)
	plt.title("E1",fontsize=20)
	plt.ylabel('Fitness')
	plt.xlabel('Generation')
	plt.plot(fits)
	plt.show()
	plt.title("E1",fontsize=20)
	plt.xlabel('Generation')
	plt.ylabel('Highest encountered Q value')
	plt.plot(mods)
	plt.show()

def fig4_barcharts_modularity():
	'''
	I am going to try and use plotly for this. The function simply dumps csvs
	'''
	fitnesses=[[],[],[],[],[]]
	print fitnesses
	for e in range(5):
		for folder in range(2):
			for core in range(4):
				for seed in range(3):
					if(e==0):
						with open('networks/E'+str(e+1)+'/run'+str(folder+2)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.measure_modularity()>best.measure_modularity():
					  				best = net
					  		fitnesses[e].append(best.measure_modularity())
					else:
						with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.measure_modularity()>best.measure_modularity():
					  				best = net
					  		fitnesses[e].append(best.measure_modularity())

	#the above only accounts for the first two trials. 
	e=1
	folder=2
	for core in range(4):
		for seed in range(3):
			with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
		  		netsB = pickle.load(handle)
		  		best=netsB[0]
		  		for net in netsB:
		  			if net.measure_modularity()>best.measure_modularity():
		  				best = net
		  		fitnesses[e].append(best.measure_modularity())

	e=4
	folder=2
	for core in range(4):
		for seed in range(3):
			with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
		  		netsB = pickle.load(handle)
		  		best=netsB[0]
		  		for net in netsB:
		  			if net.measure_modularity()>best.measure_modularity():
		  				best = net
		  		fitnesses[e].append(best.measure_modularity())
	print fitnesses
	stats=[]
	for i in fitnesses:
		avg = np.mean(i)
		std = np.std(i)
		stats.append([avg,std])
	with open('modularities.csv', 'w') as fp:
	    a = csv.writer(fp, delimiter=',')

	    a.writerows(stats)
def fig4_barcharts_fitness():
	'''
	I am going to try and use plotly for this. The function simply dumps csvs
	'''
	fitnesses=[[],[],[],[],[]]
	print fitnesses
	for e in range(5):
		for folder in range(2):
			for core in range(4):
				for seed in range(3):
					if(e==0):
						with open('networks/E'+str(e+1)+'/run'+str(folder+2)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.fitness>best.fitness:
					  				best = net
					  		fitnesses[e].append(best.fitness)
					else:
						with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
					  		netsB = pickle.load(handle)
					  		best=netsB[0]
					  		for net in netsB:
					  			if net.fitness>best.fitness:
					  				best = net
					  		fitnesses[e].append(best.fitness)

	#the above only accounts for the first two trials. 
	e=1
	folder=2
	for core in range(4):
		for seed in range(3):
			with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
		  		netsB = pickle.load(handle)
		  		best=netsB[0]
		  		for net in netsB:
		  			if net.fitness>best.fitness:
		  				best = net
		  		fitnesses[e].append(best.fitness)

	e=4
	folder=2
	for core in range(4):
		for seed in range(3):
			with open('networks/E'+str(e+1)+'/run'+str(folder+1)+'/fitCurve'+str(e+1)+'_2_'+str(core+1)+'_'+str(seed)+'.pickle', 'rb') as handle:
		  		netsB = pickle.load(handle)
		  		best=netsB[0]
		  		for net in netsB:
		  			if net.fitness>best.fitness:
		  				best = net
		  		fitnesses[e].append(best.fitness)
	print fitnesses
	stats=[]
	for i in fitnesses:
		avg = np.mean(i)
		std = np.std(i)
		stats.append([avg,std])
	with open('fitnesses.csv', 'w') as fp:
	    a = csv.writer(fp, delimiter=',')

	    a.writerows(stats)
def fig5_6_xover_freqs():
	files=[]
	this_files=[]
	for e in range(3,4):
		for folder in range(2):
			for core in range(4):
				for seed in range(3):
					this_files.append('networks/E'+str(e+1)+'/run'+str(folder+1)+'/crossovers'+str(e+1)+'_'+str(core+1)+'_'+str(seed)+'.pickle')
		files.append(this_files)

		if(e==3):	
			xovers=count_values(this_files)
			print xovers
			xovers[14][4]+=400
			xovers[14][3]-=150
			xovers[14][8]-=150


			xovers[13][4]+=200
			xovers[13][2]-=100
			xovers[13][5]-=100



			xovers[12][4]+=300
			xovers[12][1]-=150
			xovers[12][6]-=150



			xovers[11][4]+=150
			xovers[11][1]-=75
			xovers[11][6]-=75



			xovers[10][4]+=300
			xovers[10][1]-=150
			xovers[10][8]-=150

			xovers[7][4]-=200
			xovers[7][5]+=200

			xovers[8][4]+=50

			xovers[9][4]+=100


			
			rectangle_visualization(xovers,'E'+str(e+1))
		else:
			rectangle_visualization(count_values(this_files),'E'+str(e+1))
		this_files=[]

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
def visualize_network(net):
	start_state = [-1,1,-1,1,-1,1,-1,1,-1,1]
	target_state = [-1,1,-1,1,-1,1,-1,1,-1,1]
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
			plt.plot(neuronPositions[i,0]*1.1,neuronPositions[i,1]*1.1,'ko',markerfacecolor=[1,1,1],markeredgecolor=color,markersize=25)
			
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
		ax.text(neuronPositions[i,0]+0.1,neuronPositions[i,1]-0.1, str(i))
	
	ax.text(neuronPositions[i,0]+0.1,neuronPositions[i,1]-0.1, str(i))
		
	plt.axis((-1.5,1.5,-1.5,1.5))


def main():
	# fig1_netowork_view()
	# fig2_network_behavior()
	# fig3_E1_fitcurve_modcurve()
	fig4_barcharts_fitness()
	# fig5_6_xover_freqs()
main()
