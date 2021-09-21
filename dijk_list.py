import numpy as np
import time
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm


node = namedtuple('Node', ['idx', 'weight'])


def make_graph_list(weight_matrix):
    adj_list = dict()
    num_vertices = len(weight_matrix)
    for i in range(num_vertices):
        adj_list[i] = []

    for i in range(num_vertices):
        for j in range(num_vertices):
            if weight_matrix[i][j] != 0:
                adj_list[i].append(node(j, weight_matrix[i][j]))
                adj_list[j].append(node(i, weight_matrix[i][j]))

    return adj_list


def create_random_graph(num_nodes, disconnected=0.5, seed=None, graph_type=['directed', 'connected', 'complete', 'non_full']):
    np.random.seed(seed)

    if graph_type == 'directed':
        init_matrix = np.random.uniform(size=(num_nodes, num_nodes))
        init_matrix -= np.eye(num_nodes)
        adj_matrix = init_matrix >= disconnected
        init_matrix += np.eye(num_nodes)

        num_connections = np.count_nonzero(adj_matrix)
        weight_matrix = (adj_matrix != 0) * np.random.randint(1, int(0.5 * num_connections), size=adj_matrix.shape)

    elif graph_type == 'connected':
        init_matrix = np.random.uniform(size=(num_nodes, num_nodes)) + np.eye(num_nodes, k=-1) - np.eye(num_nodes)
        disconnected = (disconnected * (num_nodes**2) - num_nodes + 1)/(num_nodes-1)**2

        adj_matrix = init_matrix >= disconnected
        num_connections = np.count_nonzero(adj_matrix)
        weight_matrix = (adj_matrix != 0) * np.random.randint(1, int(0.5 * num_connections), size=adj_matrix.shape)

    elif graph_type == 'complete':
        adj_matrix = np.ones(shape=(num_nodes, num_nodes))
    
    elif graph_type == 'non_full':
        raise Exception("Not implemented.")

    return make_graph_list(weight_matrix), weight_matrix


def construct_heap(adj_list, start_node_idx):
    dist = []
    for idx in range(len(adj_list)):
        if idx == start_node_idx:
            dist.append(node(idx, 0))
        else:   
            dist.append(node(idx, np.inf))
    
    heapify(dist, 0)

    return dist


def heapify(array, start_idx):
    if (start_idx*2+1 < len(array)):
        heapify(array, start_idx*2+1)
        if (start_idx*2+2 < len(array)):
            heapify(array, start_idx*2+2)

        fixheap(array, start_idx)


def fixheap(array, start_idx):
    if start_idx*2 + 1 >= len(array):
        return

    else:
        if start_idx*2+2 < len(array):
            right_child = array[start_idx*2+2].weight
        else:
            right_child = np.inf
        left_child = array[start_idx*2+1].weight
        
        if right_child <= left_child:
            smallest_child = right_child
            child_idx = start_idx*2+2
        else:
            smallest_child = left_child
            child_idx = start_idx*2+1

        if array[start_idx].weight <= smallest_child:
            return
        else:
            array[start_idx], array[child_idx] = array[child_idx], array[start_idx]
            fixheap(array, child_idx)
            return


def create_relevant_lists(num_vertices):
    return [None] * num_vertices, [0] * num_vertices, [np.inf] * num_vertices


def create_dijk_data(num_nodes, disconnected=0.5, seed=None, graph_type='directed', start_node_idx=0):
    adj_list, weight_matrix = create_random_graph(num_nodes, disconnected, seed, graph_type)
    prev_node, traversed, distance = create_relevant_lists(num_nodes)
    priority_q = construct_heap(adj_list, start_node_idx)

    return adj_list, prev_node, traversed, priority_q, distance, weight_matrix


def remove_root(heap):
    root = heap[0]
    last_leaf = heap.pop(-1)
    if heap:
        heap[0] = last_leaf

    return root
    

def dijkstra_shortest_path(num_nodes, disconnected=0.5, seed=None, graph_type='connected', start_node_idx=0):
    adj_list, prev_node, traversed, p_queue, distance, weight_matrix = create_dijk_data(num_nodes, disconnected, seed, graph_type, start_node_idx)
    distance[start_node_idx] = 0
    traversed[start_node_idx] = 1
    
    cur_time = time.time()

    while p_queue:
        cur_node = remove_root(p_queue)
        traversed[cur_node.idx] = 1

        for vertex in adj_list[cur_node.idx]:
            original_dist = distance[vertex.idx]
            new_dist = distance[cur_node.idx] + vertex.weight

            if (not traversed[vertex.idx]) and (new_dist < original_dist):
                p_queue[p_queue.index(node(vertex.idx, original_dist))] = node(vertex.idx, new_dist)
                prev_node[vertex.idx] = cur_node.idx
                distance[vertex.idx] = new_dist

        heapify(p_queue, 0)

    time_taken =  time.time() - cur_time

    return weight_matrix, prev_node, traversed, distance, time_taken


def shortest_path(node_list, end_idx):
    path = [end_idx]
    prev = node_list[end_idx]
    while prev != None:
        path.append(prev)
        end_idx = prev
        prev = node_list[end_idx]
    
    return path[::-1]


def main():
    vertices = np.arange(20, 501, 20)
    edges = np.arange(0.1, 1, 0.1)

    total_time_vertices = []
    total_time_edges = []

    print('analysing nodes.')
    for num_nodes in tqdm(vertices):
        time_taken = 0
        for _ in range(10):
            _, _, _, _, one_loop_time = dijkstra_shortest_path(num_nodes, disconnected=0.5)
            time_taken += one_loop_time
        total_time_vertices.append(time_taken)
    
    print('analysing edges.')
    for perc_disconnected in tqdm(edges):
        time_taken = 0
        for _ in range(10):
            _, _, _, _, one_loop_time = dijkstra_shortest_path(50, disconnected=1-perc_disconnected)
            time_taken += one_loop_time
        total_time_edges.append(time_taken)
    
    fig, axes = plt.subplots(2, figsize=(15,15))
    axes[0].plot(vertices, total_time_vertices, 'tab:green')
    axes[0].set_title('Time complexity for vertices')
    axes[1].plot(edges, total_time_edges, 'tab:orange')
    axes[1].set_title('Time complexity for edges')
    plt.savefig('results.png')
    
# main()