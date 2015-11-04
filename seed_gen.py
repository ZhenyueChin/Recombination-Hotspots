#generates a set of 30 unique seed values and stores them in seed.pickle
import random
import pickle
seeds=[]

for trial in range(4):
	for i in range(10):
		seeds.append(random.random());
	pickle.dump( seeds, open( "seeds"+str(trial)+".pickle", "wb" ) )