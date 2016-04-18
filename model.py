#GRN
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

class GRN(object):
	'''
	Meant to serve as a very rudimentary model of a simple gene regulatory network,
	as a boolean network of 10 nodes with binary interactions of activation or repression 
	between nodes. 
	'''

	def __init__(self, n, max_cycles, e = [], age = 0, crossover_preference = None):
		'''
		todo: In the paper, how the initial population is formed is not entirely clear.
		are we supposed to have 100 clones of a single random network for the initial population, 
		or 100 random networks?
		'''

		self.nodes = GRN.matrix_create(max_cycles,n)
		if(crossover_preference is None):
			crossover_preference = np.random.dirichlet(np.ones(11))
			#print "no prior xover"
		self.crossover_preference=crossover_preference

		if(e==[]):
			self.edges = self.initialize_edges(len(n),len(n))
		else:
			self.edges = e
		self.fitness = -1
		self.genetic_age=age
		#print "xover pref at birth: "+str(sum(crossover_preference))+str(self.crossover_preference)
	def __str__(self):
		return "\n"+str(self.edges)+"\n"

	def copy(self):
		'''
		deep copy
		'''
		#print self.nodes
		return GRN(np.copy(self.nodes[0]),self.nodes.shape[0],np.copy(self.edges),self.genetic_age,np.copy(self.crossover_preference))

	def measure_full_modularity(self):
		'''
		Uses the Girvan-Newman algorithm to generate a 'modularity rating' for this network
		by first converting it to a networkx graph
		TODO: why on earth am I altering the GRN's actual edges? my god.
		'''

		self.edges=np.squeeze(np.asarray(self.edges))#compensating for wierd nparray vs matrix bug
		rows, cols = np.where(self.edges != 0)
		edges = zip(rows.tolist(), cols.tolist())

		gr = nx.Graph()
		gr.add_edges_from(edges)
		partition = community.best_partition(gr)
		print partition
		#partition = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
		
		return community.modularity(partition,gr)
	def measure_modularity(self):
		'''
		Uses the Girvan-Newman algorithm to generate a 'modularity rating' for this network
		by first converting it to a networkx graph
		TODO: why on earth am I altering the GRN's actual edges? my god.
		'''

		self.edges=np.squeeze(np.asarray(self.edges))#compensating for wierd nparray vs matrix bug
		rows, cols = np.where(self.edges != 0)
		edges = zip(rows.tolist(), cols.tolist())

		gr = nx.Graph()
		gr.add_edges_from(edges)
		#partition = community.best_partition(gr)
		#print partition
		partition = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10:1, 11:1}
		
		return community.modularity(partition,gr)
		#@profile

	@staticmethod
	def hamming_distance(attractor, target):
		'''
		TODO: make this cooler
		'''
		count = 0
		for i in range(len(attractor)):
			if(target[i] == attractor[i]):
				count += 1
		return len(attractor)-count

	def evaluate_network(self, max_cycle, num_targets,attractor_sets):
		'''
		Run the network until it reaches a stable attractor, or exceeds the allowed number
		of generations, return the fitness of this individual
		'''
		fitness = 0.0
		#generate a DETERMINISTIC set of perturbations of the target attractor:
		#start_attractors = generate_initial_attractors(target_attractor,200,p)
		fitness_values = list()
		target_attractors=[np.array([-1,1,-1,1,-1,1,-1,1,-1,1,-1,1]),
						   np.array([-1,1,-1,1,-1, 1,1,-1,1,-1,1,-1])]

		for set_index in range(num_targets):
		
			for initial_state in attractor_sets[set_index]:
				self.nodes=np.zeros(self.nodes.shape)
				self.nodes[0]=initial_state
				counter = 1
				while(counter < max_cycle and self.update_state(counter)):
					counter += 1

				if(counter < max_cycle):
					#stable, not chaotic or cyclic
					ham = GRN.hamming_distance(self.nodes[counter-1],target_attractors[set_index])
					#print self.nodes[counter-1]
					this_fitness = (1-(ham/float(len(target_attractors[set_index])))) #raise to the 5th
					fitness_values.append(this_fitness)
				else:
					fitness_values.append(0) #zero fitness for chaotic/cyclic state

		#print fitness_values
		tot_fitness = sum(fitness_values)
		# tot_starting_attractors = 0
		# for attractor_set in range(num_targets):
		# 	for attractor in attractor_sets[attractor_set]:
		# 		tot_starting_attractors+=1
		#tot_starting_attractors = 11*num_targets
		tot_starting_attractors = float(num_targets)*len(attractor_sets[0])
		self.fitness= tot_fitness/tot_starting_attractors
		return self.fitness

	def get_connectedness(self):
		return np.count_nonzero(self.edges)

	def crossover(net1,net2,E,core,trial_counter,gen):
		'''
		select appropriate crossover based on E, and record the crossover point
		'''
		if(E==2):
			c1,c2,x_over =  GRN.E2_naive_crossover(net1,net2)
		elif(E==3):
			c1,c2,x_over =  GRN.E3_fixed_crossover(net1,net2)
		elif(E==4):
			c1,c2,x_over =  GRN.E4_evolved_crossover(net1,net2)
		elif(E==5):
			c1,c2,x_over =  GRN.E5_modularity_crossover(net1,net2)
		else:
			print "invalid E value"

		if(E!=1):
			x_over_file = open('networks/crossovers'+str(E)+'_'+core+'_'+str(trial_counter)+'.pickle','rb')
			xover_list = pickle.load(x_over_file)
			x_over_file.close()
			xover_list.append([gen,x_over])
			x_over_file = open('networks/crossovers'+str(E)+'_'+core+'_'+str(trial_counter)+'.pickle','wb')
			pickle.dump(xover_list,x_over_file)
			x_over_file.close()

	
  			

  		return c1,c2

	def E5_get_xover_dist(self):
		'''
		get the probabilistic xover distribution based purely on modularity ratings of the 9 possible xover indexes
		'''
		#print self.edges
		self_edges=np.squeeze(np.asarray(self.edges))#compensating for wierd nparray vs matrix bug
		rows, cols = np.where(self_edges != 0)
		edges = zip(rows.tolist(), cols.tolist())

		gr = nx.Graph()
		gr.add_edges_from(edges)
		modularity_ratings=np.zeros(11)
		for i in range(1,12): #possible indexes are [1,9]
			seq1 =range(0,i)
			seq2 =range(11,i-1,-1)

			partition = dict.fromkeys(seq1,0)
			partition.update(dict.fromkeys(seq2,1))
			modularity_ratings[i-1]=(community.modularity(partition,gr))

		return GRN.normalize_to_prob_distribution(modularity_ratings)

	@staticmethod
	def normalize_to_prob_distribution(dist):
		'''
		note: only to be used when list contains negative elements, as it does un-flatten the distribution slightly
		'''
		#normalize the list to be positive
		dist+=math.fabs(min(dist))
		# and sum to one:
		return np.array([float(i)/sum(dist) for i in dist])
		

	@staticmethod
	def sample_from_prob_distribution(distribution):
		if(sum(distribution)<0.99 or sum(distribution)>1.01):
			print "Error in distribution! Does not sum to 1!"
			print distribution
			print sum(distribution)
		# make sure it sums to one:
		distribution = np.array([float(i)/sum(distribution) for i in distribution])
		choice=rand.random()
		tot=0
		x_point=0
		while(choice>tot and x_point<len(distribution)):
			tot+=distribution[x_point]
			x_point+=1

		return x_point
	@staticmethod
	def E5_modularity_crossover(net1,net2):
		'''
		Note that as of 1/7/16, the parent networks are NOT affected by this function. Two children networks are returned.
		'''
		#print "E5"
		#E5: crossover index is derived probabilistically by modularity rating
		net1.crossover_preference = net1.E5_get_xover_dist()
		net2.crossover_preference = net2.E5_get_xover_dist()
		
		new_probability_matrix=(net1.crossover_preference+net2.crossover_preference)/2 #not actually used in E5

		crossover_index = GRN.sample_from_prob_distribution(new_probability_matrix)
			
		child1=GRN(np.zeros(12),net1.nodes.shape[0],
			       np.concatenate([net2.edges[:crossover_index],net1.edges[crossover_index:]]),
				   max(net1.genetic_age,net2.genetic_age),
				   new_probability_matrix)

		child2=GRN(np.zeros(12),net1.nodes.shape[0],
			       np.concatenate([net1.edges[:crossover_index],net2.edges[crossover_index:]]),
				   max(net1.genetic_age,net2.genetic_age),
				   new_probability_matrix)

		return child1,child2,crossover_index
	@staticmethod
	def E4_evolved_crossover(net1,net2):
		'''
		Note that as of 1/7/16, the parent networks are NOT affected by this function. Two children networks are returned.
		'''
		
		#E4: crossover index is derived from an average of the two crossing-over network meta-arrays
		old_probability_matrices = [net1.crossover_preference,net2.crossover_preference]
		#print "at start: "+str(sum(net1.crossover_preference))
		new_probability_matrix=(net1.crossover_preference+net2.crossover_preference)/2.0
		#print sum(net1.crossover_preference)
		#print sum(net2.crossover_preference)
		crossover_index = GRN.sample_from_prob_distribution(new_probability_matrix)
		
		prob_mat_choice = rand.choice([0,1])
		child1=GRN(np.zeros(12),net1.nodes.shape[0],
			       np.concatenate([net2.edges[:crossover_index],net1.edges[crossover_index:]]),
				   max(net1.genetic_age,net2.genetic_age),
				   old_probability_matrices[prob_mat_choice])

		child2=GRN(np.zeros(12),net1.nodes.shape[0],
			       np.concatenate([net1.edges[:crossover_index],net2.edges[crossover_index:]]),
				   max(net1.genetic_age,net2.genetic_age),
				   old_probability_matrices[abs(prob_mat_choice-1)])
		#print child1.edges
		#print "at end: "+str(sum(net1.crossover_preference))
		#print "and children: "+str(sum(child1.crossover_preference))+" "+str(sum(child2.crossover_preference))
		return child1,child2,crossover_index

	@staticmethod
	def E3_fixed_crossover(net1,net2):
		'''
		Note that as of 1/7/16, the parent networks are NOT affected by this function. Two children networks are returned.
		'''
		crossover_index=6
		new_probability_matrix=(net1.crossover_preference+net2.crossover_preference)/2
		child1=GRN(np.zeros(12),net1.nodes.shape[0],
			       np.concatenate([net2.edges[:crossover_index],net1.edges[crossover_index:]]),
				   max(net1.genetic_age,net2.genetic_age),
				   new_probability_matrix)

		child2=GRN(np.zeros(12),net1.nodes.shape[0],
			       np.concatenate([net1.edges[:crossover_index],net2.edges[crossover_index:]]),
				   max(net1.genetic_age,net2.genetic_age),
				   new_probability_matrix)
		#print child1.edges
		return child1,child2,crossover_index

	@staticmethod
	def E2_naive_crossover(net1,net2):
		'''
		Note that as of 1/7/16, the parent networks are NOT affected by this function. Two children networks are returned.
		'''
		crossover_index = rand.randint(1,11)
		new_probability_matrix=(net1.crossover_preference+net2.crossover_preference)/2
		child1=GRN(np.zeros(12),net1.nodes.shape[0],
			       np.concatenate([net2.edges[:crossover_index],net1.edges[crossover_index:]]),
				   max(net1.genetic_age,net2.genetic_age),
				   new_probability_matrix)

		child2=GRN(np.zeros(12),net1.nodes.shape[0],
			       np.concatenate([net1.edges[:crossover_index],net2.edges[crossover_index:]]),
				   max(net1.genetic_age,net2.genetic_age),
				   new_probability_matrix)
		#print child1.edges
		return child1,child2,crossover_index


	@staticmethod
	def matrix_create(rows, first_row):
		'''
		Initializes a 2D zero np matrix
		'''
		first_row = np.squeeze(np.asarray(first_row))
		mat = np.matrix(np.zeros(rows*len(first_row)).reshape(rows,len(first_row)))
		mat[0]=first_row
		return mat
	@staticmethod
	def initialize_edges(m, n):
		'''
		mxn integer matrix of -1 (repressing) +1 (activating) or 0 (no interaction).
		The matrix will be 10x10 (not 10x9) because I am assuming recurrent connections.
		As in the original paper, 20 connections will have random interactions to start.
		'''

		
		#generate 20 unique random indexes
		#TODO: find a cooler way to do this, or at least use generator
		seen = set()
		x, y = rand.randint(0, m-1), rand.randint(0, n-1)
		seen.add((x, y))

		while(len(seen)<20):
			while (x, y) in seen:
				x, y = rand.randint(0, m-1), rand.randint(0, n-1)
			seen.add((x,y))

		#initialize matrix
		e = GRN.matrix_create(m, np.zeros(n))

	    #apply random interactions to unique indexes
		for pair in seen:
			if (rand.random()<0.5):
				e[pair[0],pair[1]] = 1
			else: 
				e[pair[0],pair[1]] = -1

		return e

	def perturb(self, mu, E):
		'''
		Each gene (node) has a chance of mu to mutate
		'''
		for i in range(self.nodes.shape[1]):
			if(rand.random()<=mu):
				#print "mutating gene "+str(i)+"!"
				self.mutate(i)

		#for now, this also holds true for xover preference:
		if(E==4):
			for i in range(len(self.crossover_preference)):
				if(rand.random()<=mu):
					self.mutate_meta_xover(i)

	def mutate(self,i):
		'''
		Mutates the specified gene at index i according to the rule specified in page 9 of the original paper.
		Also mutates the prefered crossover point metainformation
		'''
		N = self.nodes.shape[1]									#number of nodes in the network
		r_u = 0  												#number of regulators for this gene
		for j in range(0,self.edges.shape[0]):
			if(self.edges[j,i]!=0):  							#if the mutated node is regulated by j
				r_u+=1

		probability_to_lose_interaction=(4.0*r_u)/(4*r_u+(N-r_u))  #check this formula with Dr. Bongard, it doesn't make sense
		
		if(rand.random() <= probability_to_lose_interaction):
			#lose an interaction
			interactions = []
			for edge in range(0,self.edges.shape[1]):
				if(edge != 0):
					interactions.append(edge)
			if(len(interactions) > 0):
				toRemove = rand.choice(interactions)
				self.edges[toRemove,i]=0
				#print "removed edge from "+str(toRemove)+" to "+str(i)
				#raw_input("enter to continue")
		else:
			#gain an interaction
			non_interactions=[]
			for edge in range(0,self.edges.shape[1]):
				if(self.edges[edge,i]==0):
					non_interactions.append(edge)
			if(len(non_interactions)>0):
				toAdd=rand.choice(non_interactions)
				if(rand.random()<0.5):
					self.edges[toAdd,i]=1
					#print "added positive edge from "+str(toAdd)+" to "+str(i)
				else:
					self.edges[toAdd,i]=-1
					#print "added negative edge from "+str(toAdd)+" to "+str(i)
				
		

	def mutate_meta_xover(self,src_index):
		'''
		mutates the metainformation array dictating xover probability in different indexes.
		'''
		oldtot=math.fsum(self.crossover_preference)
		#choose magnitude of mutation from random gaussian
		self.crossover_preference[src_index] = math.fabs(rand.gauss(self.crossover_preference[src_index],
													      math.fabs(self.crossover_preference[src_index])))
		#print "new value: "+str(self.crossover_preference[src_index])
		#re-normalize the matrix
		total = math.fsum(self.crossover_preference)
		self.crossover_preference = np.array([float(i)/total for i in self.crossover_preference])
		newtot=math.fsum(self.crossover_preference)
		if(oldtot-newtot>0.01 or oldtot-newtot<-0.01):
			print "mutated wrong! new and old tots:"
			print newtot
			print oldtot
			print "crossover_preference:"
			print self.crossover_preference
	#@profile
	def update_state(self,t):
		'''
		Runs one iteration of network interactions, a single timestep
		As described on page 2 of the original paper, in 'Model'
		Returns a boolean: false if there is no change in network state (has reached a 
		stable attractor).
		'''

		#efficient matrix manipulation, mis-represents nodes with no inputs:
		self.nodes[t,:] = np.matrix(self.nodes[t-1,:])*self.edges
		self.nodes[t,:] = np.clip(self.nodes[t,:],-1,1)
		self.nodes[t,:][self.nodes[t,:] == 0] = -1

		#check for nodes with no inputs:
		for target_gene in range(0,12):
			if(np.count_nonzero(self.edges[:,target_gene])==0):
				self.nodes[t,target_gene]=self.nodes[t-1,target_gene]

		return (not np.array_equal(self.nodes[t-1,:],self.nodes[t,:]))



	def rectangle_visualization(self,initial_states,target, title):
		'''
		Shows the network behavior by timestep, with target
		'''
		print "Rectangle Visualization"

		num_columns=int((math.sqrt(len(initial_states))))

		print len(initial_states)
		print num_columns
		plt.title(title,fontsize=20)
		temp = self.nodes
		for i in range(len(initial_states)):
			plt.subplot(len(initial_states)/num_columns+1,num_columns,i+1)
			shape = (12,self.nodes.shape[1])
			self.nodes=np.zeros(shape)
			self.nodes[0]=initial_states[i]
			counter = 1
			while(counter < self.nodes.shape[0]):
				self.update_state(counter)
				counter += 1
			plt.imshow(np.append(self.nodes,[target],axis=0), cmap=plt.cm.gray, aspect='auto',interpolation='nearest')
			plt.gca().axes.get_xaxis().set_visible(False)
			if(i% (len(initial_states)/(num_columns+1)) != 0):
				plt.gca().axes.get_yaxis().set_visible(False)
		self.nodes=temp
		plt.subplot(len(initial_states)/num_columns+1,num_columns,2)
		
		plt.show()

	def shorten_line(self,x,y):
		'''
		Does some nasty geometry to find proper x' , y' to shorten arrows appropriately in graphical representation of network
		'''
		if(y==0):
			return 0.001,0.001
		a = 0.18
		y_prime = (math.sqrt((a**2)*(y**2)-2*a*(y**2)*math.sqrt((x**2)+(y**2))+(x**2)*(y**2)+y**4))/math.sqrt((x**2)+(y**2))
		if(y<0):
			y_prime=-y_prime
		
		x_prime= ((x*1.0)/y)*y_prime
		return x_prime,y_prime

	def visualize_network(self,start_state,target_state,max_cycle):
		'''
		Shows the network as a node-edge graph over time
		'''
		self.nodes=np.zeros(self.nodes.shape)
		self.nodes[0]=start_state
		# print self.nodes
		active_nodes = []
		inactive_nodes = []
		numNeurons = self.nodes.shape[1]

		neuronPositions=self.matrix_create(numNeurons,np.zeros(2))
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
		plt.ion()
		plt.show()
		active=True
		while(active):
			
			plt.cla()
			#draw straight connections (non-reccurent)	
			for i in range(0,numNeurons):
				for other in range(0,neuronPositions.shape[0]):
					#w = int(10*abs(synapseValues[i,other]))+1
					if(self.edges[i,other]!=0):
					
						if self.edges[i,other]>0:
							color = "green"
						else:
							color = "red"
						arrow_x,arrow_y=self.shorten_line((neuronPositions[other,0]-neuronPositions[i,0]),
								(neuronPositions[other,1]-neuronPositions[i,1]))
						ax.arrow(neuronPositions[i,0],
								neuronPositions[i,1], 
								arrow_x,arrow_y,
								head_width=0.05, head_length=0.1, fc=color, ec=color)
								#shape='left')

			for i in range(0,numNeurons):
				if(self.edges[i,i]!=0): #recurrent connection
					if self.edges[i,i]>0:
						color = "green"
					else:
						color = "red"
					plt.plot(neuronPositions[i,0]*1.1,neuronPositions[i,1]*1.1,'ko',markerfacecolor=[1,1,1],markeredgecolor=color,markersize=25)
				
				#target state:
				if(target_state[i]==1):		
					active_nodes += plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[0,0,0],markersize=20)
				else:
					inactive_nodes += plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[1,1,1],markersize=20)
				#true neuron state:
				if(self.nodes[counter,i]==1):		
					active_nodes += plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[0,0,0],markersize=16)
				else:
					inactive_nodes += plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[1,1,1],markersize=16)
				ax.text(neuronPositions[i,0]+0.1,neuronPositions[i,1]-0.1, str(i))
			
			counter += 1
			plt.axis((-1.5,1.5,-1.5,1.5))
			plt.draw()
			time.sleep(0.5)
			"still visualizing"
			active = (counter < max_cycle and self.update_state(counter))
		# print self.nodes
		

# def main():
# 	grn = GRN(np.matrix([-1,1,-1,1,-1,1,-1,1,-1,1]))
# 	for i in range(0,100):
# 		grn.update_state()
# 		#print grn.edges
# main()