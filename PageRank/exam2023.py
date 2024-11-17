from main import Node, page_rank


# Example usage
# Create nodes
A = Node("A")
B = Node("B")
C = Node("C")
D = Node("D")
E = Node("E")
F = Node("F")

# Define relationships (e.g., A links to B and C, B links to C and D, etc.)
A.add_incoming(B)
A.add_incoming(C)
A.add_outgoing(F)

B.add_incoming(F)
B.add_incoming(C)
B.add_outgoing(A)
B.add_outgoing(C)
B.add_outgoing(E)

C.add_incoming(B)
C.add_incoming(C)
C.add_outgoing(A)
C.add_outgoing(B)
C.add_outgoing(C)
C.add_outgoing(D)
C.add_outgoing(E)
C.add_outgoing(F)

D.add_incoming(D)
D.add_incoming(C)
D.add_outgoing(D)

E.add_incoming(B)
E.add_incoming(F)
E.add_incoming(C)
E.add_incoming(E)
E.add_outgoing(E)

F.add_incoming(A)
F.add_incoming(C)
F.add_outgoing(E)
F.add_outgoing(B)



# Add nodes to a list
nodes = [A, B, C, D, E, F]

# Calculate PageRank values
iter = 2
damp = 0.2
rounding = 3
result = page_rank(nodes, iter, damp, rounding)