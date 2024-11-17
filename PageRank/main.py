class Node:
    def __init__(self, name):
        self.name = name
        self.incoming = []
        self.outgoing = []

    def add_incoming(self, node):
        self.incoming.append(node)

    def add_outgoing(self, node):
        self.outgoing.append(node)


def page_rank(nodes, iterations, damping_factor, rounder=3):
    num_nodes = len(nodes)
    initial_rank = round(1 / num_nodes, rounder)
    ranks = {node.name: initial_rank for node in nodes}

    
    for i in range(iterations):
        new_ranks = {}
        
        for node in nodes:
            incoming_ranks = []
            for node1 in node.incoming:
                # print("INCOMING NAME: ", node1.name)
                # print("RANK: ", ranks[node1.name])
                # print("LEN OUTGOING: ", len(node1.outgoing))
                incoming_ranks.append(ranks[node1.name] / len(node1.outgoing))
                # print(incoming_ranks)

            incoming_rank_sum = sum(incoming_ranks)
            # print(F"INCOMING RANK: {incoming_rank_sum}")
            new_ranks[node.name] = round((damping_factor / num_nodes) + ((1 - damping_factor) * incoming_rank_sum), rounder)
        
        ranks = new_ranks  # Update ranks for the next iteration
        print(sum(list(ranks.values())))
        print(f"Iteration: {i + 1}, Ranks: {ranks}")

    return ranks



if __name__ == "__main__":
    A = Node("A")
    B = Node("B")
    C = Node("C")

    A.add_incoming(C)
    A.add_outgoing(C)
    A.add_outgoing(B)

    B.add_outgoing(C)
    B.add_incoming(A)

    C.add_incoming(A)
    C.add_incoming(B)
    C.add_outgoing(A)

    nodes = [A, B, C]
    iterations = 2
    damping = 0.2
    rounding = 2
    page_rank(nodes, iterations, damping, rounding)

    print(f"################## {len(C.incoming)}")

