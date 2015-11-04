import sys
import thread
import subprocess
import threading
import Queue

def call_sim(q,seedsfile, outfile1, outfile2, num_targets):
	q.put(subprocess.call("python pareto_k_tournament.py "+seedsfile+" "+outfile1+" "+outfile2+" "+num_targets,shell=True))

q = Queue.Queue


for i in range(4):
	t = threading.Thread(target=call_sim,args = (q,"seeds"+str(i),"out"+str(i)+"_1","out"+str(i)+"_2",str(i+2)))
	t.daemon=True
	t.start()


print "subprocesses launched"	