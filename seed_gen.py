#generates a set of 30 unique seed values and stores them in seed.pickle
import random
import pickle
seeds=[]
for i in range(30):
	seeds.append(random.random());

pickle.dump( seeds, open( "seeds.pickle", "wb" ) )