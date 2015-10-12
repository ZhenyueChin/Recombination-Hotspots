# Specialization-Modularity

This is my reimplementation of the work of Espinosa-Soto and Wagner in their 2010 paper Specialization Can Drive the Evolution of Modularity. The implementation is in python, and completed under the supervision of Dr. Bongard at the University of Vermont as part of an undergraduate research project, Fall 2015

For modularity scoring we usethe louvain 0.5.3 package. Documentation is available here: https://pypi.python.org/pypi/louvain


Documentation for community Modularity scoring, from the online documentation:

* Modularity.
This method compares the actual graph to the expected graph, taking into
account the degree of the nodes [1]. The expected graph is based on a
configuration null-model. Notice that we use the non-normalized version (i.e.
we don't divide by the number of edges), so that this Modularity values
generally does not fall between 0 and 1. The formal definition is

```
H = sum_ij (A_ij - k_i k_j / 2m) d(s_i, s_j),
```

where `A_ij = 1` if there is an edge between node `i` and `j`, `k_i` is the degree of
node `i` and `s_i` is the community of node i.
