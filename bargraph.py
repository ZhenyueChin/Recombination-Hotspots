import csv
import numpy as np
import matplotlib.pyplot as plt
def get_values(filename):
	with open(filename+'.csv', 'rb') as f:
	    reader = csv.reader(f)
	    fits = list(reader)

	return fits

def graph_fits():
	fits = get_values('fitnesses')

	N = 2
	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, [float(f[0]) for f in fits], width, color='r', yerr=[float(f[1]) for f in fits])


	#rects2 = ax.bar(ind + width, [float(m[0]) for m in mods], width, color='y', yerr=[float(m[1]) for m in mods])

	# add some text for labels, title and axes ticks
	ax.set_title('Average Final Error')
	ax.set_ylabel('Normalized Error')
	ax.set_xticks(ind + width)
	ax.set_xticklabels(('E4', 'E5'))

	axes = plt.gca()
	axes.set_ylim([0,.1])
	plt.autoscale()
	# ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
	#ax.legend((rects1[0]), ('Fitness'))
	#autolabel(rects1,ax)
	#autolabel(rects2)

	plt.show()
def graph_mods():
	fits = get_values('modularities')
	
	N = 2
	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, [float(f[0]) for f in fits], width, color='r', yerr=[float(f[1]) for f in fits])


	#rects2 = ax.bar(ind + width, [float(m[0]) for m in mods], width, color='y', yerr=[float(m[1]) for m in mods])

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Q5 Score')
	ax.set_title('Average Modularity of Final Networks')
	ax.set_xticks(ind + width)
	ax.set_xticklabels(('E4', 'E5'))

	axes = plt.gca()
	axes.set_ylim([0,.5])
	plt.autoscale()
	# ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
	#ax.legend((rects1[0]), ('Fitness'))
	#autolabel(rects1,ax)
	#autolabel(rects2)

	plt.show()
def autolabel(rects,ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')



def main():
	graph_fits()
	graph_mods()

main()
