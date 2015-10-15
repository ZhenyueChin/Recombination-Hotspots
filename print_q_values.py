
import pickle
with open('q_values.pickle', 'rb') as handle:
	vals = pickle.load(handle)
print vals