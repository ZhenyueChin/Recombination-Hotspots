#run a t-test on two sets of values
import pickle
from scipy import stats
import numpy

file1 = 'networks/populationsA.pickle'
file2 = 'seeds1.pickle'

with open(file1, 'rb') as handle:
	vals1 = pickle.load(handle)
with open(file2, 'rb') as handle:
	vals2 = pickle.load(handle)

t_val = stats.ttest_ind(vals1,vals2)
print t_val
print "--------------------"
print "avg of vals1: "+str(numpy.mean(vals1))
print "avg of vals2: "+str(numpy.mean(vals2))
print "--------------------"
print vals1
print vals2