#GRN
from scipy import *
import random as rand
import numpy as np
import matplotlib.pyplot as plt

class GRN(object):
	'''
	Meant to serve as a very rudimentary model of a simple gene regulatory network,
	as a boolean network of 10 nodes with binary interactions of activation or repression 
	between nodes. 
	'''

	def __init__(self, n, max_cycles, e):
		'''
		todo: In the paper, how the initial population is formed is not entirely clear.
		are we supposed to have 100 clones of a single random network for the initial population, 
		or 100 random networks?
		'''
		self.nodes = GRN.matrix_create(max_cycles,n)
		self.edges = e
		self.fitness = -1

	def __str__(self):
		return "\n"+str(self.edges)+"\n"

	def copy(self):
		'''
		deep copy
		'''
		#print type(GRN(len(self.nodes),self.edges.copy()))
		return GRN(np.copy(self.nodes[0]),self.nodes.shape[0],np.copy(self.edges))

	def measure_modularity(self):
		'''
		Uses the Girvan-Newman algorithm to generate a 'modularity rating' for this network
		'''
		
	@staticmethod
	def matrix_create(rows, first_row):
		'''
		Initializes a 2D zero np matrix
		'''
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

	def perturb(self, mu):
		'''
		Each gene (node) has a chance of mu to mutate
		'''
		for i in range(self.nodes.shape[1]):
			if(rand.random()<=mu):
				#print "mutating gene "+str(i)+"!"
				self.mutate(i)

	def mutate(self,i):
		'''
		Mutates the specified gene at index i according to the rule specified in page 9 of the original paper
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
				
				#raw_input("enter to continue")

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
		for target_gene in range(0,10):
			if(np.count_nonzero(self.edges[:,target_gene])==0):
				self.nodes[t,target_gene]=self.nodes[t-1,target_gene]

		return (not np.array_equal(self.nodes[t-1,:],self.nodes[t,:]))

	def visualize_state(self):
		'''
		Shows the current activation state of the 10 genes
		'''

		plt.imshow(np.reshape(self.nodes, (-1, 1)), cmap=plt.cm.gray, aspect='auto',interpolation='nearest',origin='left')
		plt.show()
	def shorten_line(self,x,y):
		'''
		Does some nasty geometry to find proper x' , y' to shorten arrows appropriately in graphical representation of network
		'''
		if(y==0 and x==0):
			return 0,0
		a = 0.18
		y_prime = (math.sqrt((a**2)*(y**2)-2*a*(y**2)*math.sqrt((x**2)+(y**2))+(x**2)*(y**2)+y**4))/math.sqrt((x**2)+(y**2))
		if(y<0):
			y_prime=-y_prime
		x_prime= ((x*1.0)/y)*y_prime
		return x_prime,y_prime

	def visualize_network(self):
		'''
		Shows the network as a node-edge graph
		'''
		active_nodes = []
		inactive_nodes = []
		numNeurons = self.nodes.shape[0]
		neuronPositions=self.matrix_create(numNeurons,np.zeros(2))
		#compute positions of neurons for the circular visualization
		angle = 0.0
		angleUpdate = 2 * pi /numNeurons
		ax = plt.axes()
		for i in range(0,numNeurons):
			x = sin(angle)
			y = cos(angle)
			angle = angle + angleUpdate
			neuronPositions[i,0]=x
			neuronPositions[i,1]=y

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
			if(self.nodes[i]==1):
					
				active_nodes += plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[0,0,0],markersize=18)
			else:
					
				inactive_nodes += plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[1,1,1],markersize=18)
			ax.text(neuronPositions[i,0]+0.1,neuronPositions[i,1]-0.1, str(i))
		#plt.legend()
		plt.axis((-1.5,1.5,-1.5,1.5))
		plt.show()
		

# def main():
# 	grn = GRN(np.matrix([-1,1,-1,1,-1,1,-1,1,-1,1]))
# 	for i in range(0,100):
# 		grn.update_state()
# 		#print grn.edges
# main()