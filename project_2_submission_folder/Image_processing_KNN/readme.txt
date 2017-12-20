Explanation of Exercise 10:

Set-up Laplace-Beltrami:

1) Iterate over all vertices;
2) For a single vertex v_i get two Halfedge_around_vertex_circulator to scan the neighbours v_j ;
3) Being “e” the edge that connect v_i and v_j get the weight of this edge , cotan[e]. Store this value on the (i,j) position of L.
4) After having iterate over all neighbors store  opposite of the sum of weights of the neighbors in the position (i,i) of L 





Distribution of work:
