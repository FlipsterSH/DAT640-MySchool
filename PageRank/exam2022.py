from main import Node, page_rank



if __name__ == "__main__":
    A = Node("A")
    B = Node("B")
    C = Node("C")
    D = Node("D")

    A.add_incoming(B)
    A.add_incoming(C)
    A.add_outgoing(D)

    B.add_outgoing(A)
    B.add_outgoing(C)
    B.add_incoming(D)

    C.add_incoming(B)
    C.add_incoming(D)
    C.add_outgoing(A)

    D.add_incoming(A)
    D.add_outgoing(B)
    D.add_outgoing(C)

    nodes = [A, B, C, D]
    iterations = 2
    damping = 0.2
    rounding = 2
    page_rank(nodes, iterations, damping, rounding)

