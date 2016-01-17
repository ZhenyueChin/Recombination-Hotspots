


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

attractor_setsA = generate_permutations([-1,1,-1,1,-1,1,-1,1,-1,1])
attractor_setsB = generate_permutations([-1,1,-1,1,-1,-1,1,-1,1,-1])
for i in range(len(attractor_setsA)):
	for j in range(len(attractor_setsB)):
		if(attractor_setsA[i]==attractor_setsB[j]):
			print attractor_setsA[i]
			print i
			print j
print len(attractor_setsA)
print len(attractor_setsB)
a = [generate_permutations([-1,1,-1,1,-1,1,-1,1,-1,1]),generate_permutations([-1,1,-1,1,-1,1,-1,1,-1,1])]
print a
print len(a)
print len(a[0])