#run a t-test on two sets of values
import pickle
from scipy import stats

file1 = 'q_values_single.pickle'
file2 = 'q_values_two.pickle'

with open(file1, 'rb') as handle:
	vals1 = pickle.load(handle)
with open(file2, 'rb') as handle:
	vals2 = pickle.load(handle)

t_val = stats.ttest_ind(vals1,vals2)
print t_val
print "--------------------"
print vals1
print vals2