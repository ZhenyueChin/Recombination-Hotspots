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

	def __init__(self, n, e):
		'''
		todo: In the paper, how the initial population is formed is not entirely clear.
		are we supposed to have 100 clones of a single random network for the initial population, 
		or 100 random networks?
		'''

		self.nodes = n
		self.edges = e
		self.fitness = -1

	def __str__(self):
		return "\n"+str(self.edges)+"\n"

	def copy(self):
		'''
		deep copy
		'''
		#print type(GRN(len(self.nodes),self.edges.copy()))
		return GRN(self.nodes,self.edges.copy())

	@staticmethod
	def matrix_create(rows, columns):
		'''
		Initializes a 2D zero np matrix
		'''

		return np.matrix(np.zeros(rows*columns).reshape(rows,columns))

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
		e = GRN.matrix_create(m, n)

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
		for i in range(self.nodes.shape[0]):
			if(rand.random()<=mu):
				#print "mutating gene "+str(i)+"!"
				self.mutate(i)

	def mutate(self,i):
		'''
		Mutates the specified gene at index i according to the rule specified in page 9 of the original paper
		'''
		N = self.nodes.shape[0]									#number of nodes in the network
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
				toRemove = random.choice(interactions)
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
				toAdd=random.choice(non_interactions)
				if(rand.random()<0.5):
					self.edges[toAdd,i]=1
					#print "added positive edge from "+str(toAdd)+" to "+str(i)
				else:
					self.edges[toAdd,i]=-1
					#print "added negative edge from "+str(toAdd)+" to "+str(i)
				
				#raw_input("enter to continue")

	def update_state(self):
		'''
		Runs one iteration of network interactions, a single timestep
		As described on page 2 of the original paper, in 'Model'
		Returns a boolean, true if there is no change in network state: has reached a 
		stable attractor.
		Accepts a maximum number of iterations (to handle possible chaotic states)
		'''

		counter = 0
		still_active = False
		influence = 0
		old_nodes = self.nodes.copy()
		for regulated_node in range(self.edges.shape[0]):
			for regulator_node in range(self.edges.shape[1]):
				influence += (self.edges[regulator_node,regulated_node])*old_nodes[regulator_node]

			if influence > 0:
				self.nodes[regulated_node] = 1
			else:
				self.nodes[regulated_node] = -1
			if(old_nodes[regulated_node]!=self.nodes[regulated_node]):
				still_active=True

			#print old_nodes
			#print self.nodes
		return still_active

	def visualize_state(self):
		'''
		Shows the current activation state of the 10 genes
		'''

		plt.imshow(np.reshape(self.nodes, (-1, 1)), cmap=plt.cm.gray, aspect='auto',interpolation='nearest',origin='left')
		plt.show()

	def visualize_network(self):
		'''
		Shows the network as a node-edge graph
		'''
		active_nodes = []
		inactive_nodes = []
		numNeurons = len(self.nodes)
		neuronPositions=matrix_create(numNeurons,2)
		#compute positions of neurons for the circular visualization
		angle = 0.0
		angleUpdate = 2 * pi /numNeurons
		for i in range(0,numNeurons):
			x = sin(angle)
			y = cos(angle)
			angle = angle + angleUpdate
			neuronPositions[i,0]=x
			neuronPositions[i,1]=y
			if(self.nodes[i]==1):
				active_nodes += plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[0,0,0],markersize=18)
			else:
				inactive_nodes += plt.plot(neuronPositions[i,0],neuronPositions[i,1],'ko',markerfacecolor=[1,1,1],markersize=18)
		
		for i in range(0,numNeurons):
			for other in range(0,neuronPositions.shape[0]):
				#w = int(10*abs(synapseValues[i,other]))+1
				if(self.edges[i,other]!=0):
				
					if self.edges[i,other]>0:
						color = "green"
					else:
						color = "red"
		
					plt.plot([neuronPositions[i,0],neuronPositions[other,0]],
						[neuronPositions[i,1],neuronPositions[other,1]],color)
		plt.legend()
		plt.show()
		

# def main():
# 	grn = GRN(np.array([-1,1,-1,1,-1,1,-1,1,-1,1]))
# 	for i in range(0,100):
# 		grn.update_state()
# 		#print grn.edges
# main()