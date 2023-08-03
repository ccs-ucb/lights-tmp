# coding: utf-8
# wdt@berkeley.edu

import click
import numpy as np
import json
import networkx as nx
import random
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import pygraphviz as pgv
import pandas as pd

colors_rgb = [
    {"r":253,"g":181, "b":21},
    {"r":238,"g":31, "b":96},
    {"r":0,"g":176, "b":218},
    {"r":133,"g":148, "b":56},
    {"r":237,"g":78, "b":51}, 
    {"r":221,"g":213, "b":199},  
]

colors_hex = [
    "#%02x%02x%02x" % tuple(c.values())
    for c in colors_rgb
]


def generate_dynamic(groups, trigger_groups, target_groups):
    """This is a historical func that i might use for more complex problems later"""
    condition_groups = [0]
    source = np.isin(groups, trigger_groups).astype(int).tolist()
    target = np.isin(groups, target_groups).astype(int).tolist()
    conditions = {
        0: {
        "domain": np.isin(groups, [2]).astype(int).tolist(),
        "minsum": 1 
        }
    }
    return {
        "source": source,
        "target": target,
        "conditions": conditions
    }

def add_path_to_tree(tree, path, tree_index, root_index):
    current = tree
    for depth, node in enumerate(path):
        # Incorporate the tree index into the node representation if there is divergence
        key = (root_index, node)
        if key not in current and depth > 0:
            key = (root_index, node, tree_index)
        current = current.setdefault(key, {})

def add_edges_from_forest(graph, forest):
    for tree_root, subtree in forest.items():
        graph.add_node(tree_root, label=tree_root[1])
        add_edges_from_tree(graph, subtree, parent=tree_root)

def add_edges_from_tree(graph, tree, parent=None):
    for node, subtree in tree.items():
        graph.add_node(node, label=node[1])
        if parent is not None:
            graph.add_edge(parent, node)
        add_edges_from_tree(graph, subtree, parent=node)

def visualize_paths_as_tree(paths, colors_hex):
    # Convert list of paths into a forest (collection of trees)
    forest = {}
    root_indices = {}
    for tree_index, path in enumerate(paths):
        root = path[0]
        root_index = root_indices.setdefault(root, len(root_indices))
        subtree = forest.setdefault((root_index, root), {})
        add_path_to_tree(subtree, path[1:], tree_index, root_index)

    # Create a directed graph
    graph = nx.DiGraph()

    # Add edges to the graph from the forest
    add_edges_from_forest(graph, forest)

    # Set node labels
    labels = nx.get_node_attributes(graph, 'label')
    
    # Generate colors for each node
    node_colors = [colors_hex[label % len(colors_hex)] for label in labels.values()]

    # Draw the graph using Graphviz layout
    pos = graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_labels=False, arrows=True, width=2, node_size=500, node_color=node_colors, edge_color="black")
    nx.draw_networkx_labels(graph, pos, labels=labels)
    
    # Show the plot
    plt.show()
    
def draw_dag(DAG):
    n = len(DAG.nodes())
    pos = graphviz_layout(DAG, prog='neato')
    nx.draw(
        DAG,
        pos,
        with_labels=True,
        node_size=1000,
        node_color=colors_hex[:n],
        width=0.8,
        font_size=14,
    )
    plt.show()

def all_topological_sorts(edges):
    # Creating an adjacency list
    graph = {}
    in_degree = {}
    
    for u, v in edges:
        if u not in graph:
            graph[u] = []
            in_degree[u] = 0
        if v not in graph:
            graph[v] = []
            in_degree[v] = 0
        
        graph[u].append(v)
        in_degree[v] += 1

    # Function to perform DFS
    def dfs(graph, in_degree, stack, result):
        # Check if all nodes are in stack, i.e., a valid topological sort
        if len(stack) == len(graph):
            result.append(stack[:])
            return

        # Iterate over the nodes in the graph
        for u in graph.keys():
            # Check if in_degree is 0 and not already in the stack
            if in_degree[u] == 0 and u not in stack:
                # Decrease in_degree for the adjacent vertices
                for v in graph[u]:
                    in_degree[v] -= 1
                # Add current node to stack
                stack.append(u)
                # Recur
                dfs(graph, in_degree, stack, result)
                # Pop the current node from stack and reset in_degree
                stack.pop()
                for v in graph[u]:
                    in_degree[v] += 1

    # Placeholder to store the results
    result = []
    # Call the DFS function
    dfs(graph, in_degree, [], result)
    return result

import random

def remove_symmetric_edges(edges):
    edges_set = set(edges)
    new_edges = set()
    edges_to_remove = set()

    for edge in edges_set:
        u, v = edge

        if (edge in new_edges) or (v, u) in new_edges:
            continue
        
        if (v, u) in edges_set:
            chosen_edge = random.choice([(u, v), (v, u)])
            new_edges.add(chosen_edge)
        
        else:
            new_edges.add(edge)
    return list(new_edges)

def find_branch_length_counts(root_node):
    # Convert the node tree to a NetworkX graph
    graph = nx.DiGraph()

    def add_nodes_and_edges(node, parent=None):
        graph.add_node(node)

        if parent is not None:
            graph.add_edge(parent, node)

        for child in node.children:
            add_nodes_and_edges(child, parent=node)

    add_nodes_and_edges(root_node)

    # Find the branch lengths from root_node to leaf nodes using NetworkX
    branch_length_counts = {}

    def dfs(node, path_length):
        if graph.out_degree(node) == 0:  # Check if it's a leaf node
            if path_length in branch_length_counts:
                branch_length_counts[path_length] += 1
            else:
                branch_length_counts[path_length] = 1
            return

        for child in graph.successors(node):
            dfs(child, path_length + 1)

    dfs(root_node, 0)

    return branch_length_counts

def visualize_paths(paths):
    combined_graph = nx.DiGraph()
    
    for path in paths:
        path_graph = nx.DiGraph()
        for i in range(len(path) - 1):
            path_graph.add_edge(path[i], path[i + 1])
        combined_graph = nx.compose(combined_graph, path_graph)
    
    plt.figure(figsize=(8, 8))
    plt.title("Tree Network")
    # pos = nx.spring_layout(combined_graph)
    pos = nx.drawing.nx_agraph.graphviz_layout(combined_graph, prog='dot', args='-Gnodesep=2 -Grankdir=TB')
    
    root_node = next(node for node in combined_graph.nodes() if combined_graph.in_degree(node) == 0)
    sorted_nodes = sorted(combined_graph.nodes(), key=lambda node: len(nx.shortest_path(combined_graph, root_node, node)))
    labels = {node: node.action for node in sorted_nodes}
    node_colors = [colors_hex[label % len(colors_hex)] if label is not None else 'grey' for label in labels.values()]
    nx.draw(combined_graph, pos, nodelist=sorted_nodes, with_labels=True, labels=labels, node_size=250, node_color=node_colors, edge_color='gray', arrows=True)
    plt.axis('off')
    plt.show()

def collect_root_node_with_length(root_node, target_length):
    paths = []

    def dfs(node, current_path):
        current_path.append(node)

        if len(current_path) <= target_length and not node.children:  # Check if the current path has the target length and ends at a leaf node
            paths.append(list(current_path))
            current_path.pop()
            return

        for child in node.children:
            dfs(child, current_path)

        current_path.pop()

    dfs(root_node, [])
    return paths


def convert_tree_to_networkx(node):
    graph = nx.DiGraph()
    
    # Recursive function to add nodes and edges to the graph
    def add_nodes_and_edges(node, parent=None):
        graph.add_node(node)
        
        if parent is not None:
            graph.add_edge(parent, node)
        
        for child in node.children:
            add_nodes_and_edges(child, parent=node)
    
    # Start the recursive function
    add_nodes_and_edges(node)
    
    return graph

def plot_trees(root_node):
    # Assuming `root_node` is the root of the node trees
    tree_graph = convert_tree_to_networkx(root_node)

    # Visualize the tree network using NetworkX
    # pos = nx.spring_layout(tree_graph)
    # pos = graphviz_layout(tree_graph, prog='dot')
    pos = nx.drawing.nx_agraph.graphviz_layout(tree_graph, prog='dot')
    nx.draw(tree_graph, pos=pos, with_labels=False, node_size=10, node_color="lightblue", edge_color="gray", arrows=True)

    plt.title("Rollout Tree Network")
    plt.show()


class Node:
    def __init__(self, graph_state, knowledge_state, action, new_knowledge, depth):
        self.graph_state = graph_state
        self.knowledge_state = knowledge_state
        self.action = action
        self.new_knowledge = new_knowledge
        self.children = []
        self.depth = depth

def select_actions(graph_state, knowledge_state):
    uncertain_incoming = ((knowledge_state == 0).sum(axis=1) > 0)
    uncertain_outgoing = ((knowledge_state == 0).sum(axis=0) > 0)
    
    possible_actions = np.arange(len(graph_state))
    valued = np.logical_or(uncertain_incoming, uncertain_outgoing)
    choices = possible_actions[valued]
    return choices

def perform_action(graph_state, knowledge_state, action, adjacency):

    would_turn_off = graph_state.astype(bool) & (adjacency[action]).astype(bool)
    would_remain_on = graph_state.astype(bool) & ~(adjacency[action]).astype(bool)
    
    new_knowledge = np.zeros(knowledge_state.shape)
    newly_revealed_edges = np.logical_and(
        (knowledge_state[action] == 0),  # didn't already know 
        np.logical_or(would_turn_off, would_remain_on) # revealed by selection
    )
    new_knowledge[action][newly_revealed_edges] = 1

    newly_implied_edges = ((knowledge_state[action] == 0) & would_turn_off).astype(int)
    n = newly_implied_edges.sum()
    if n > 0:
        new_knowledge[
            np.nonzero(newly_implied_edges),
            (np.ones(n) * action).astype(int)
        ] = -1
        # new_knowledge[list(newly_implied_edges)] = 1

    # update the knowledge state to reflect anything revealed by action
    next_knowledge_state = knowledge_state.copy()
    next_knowledge_state[action][would_turn_off] = 1
    next_knowledge_state[action][would_remain_on] = -1
    next_knowledge_state[would_turn_off.astype(bool), (np.arange(len(graph_state)) == action).astype(bool)] = -1
    
    # turn on/off any lights affected by action
    next_graph_state = graph_state.copy()
    next_graph_state[action] = 1
    next_graph_state[(adjacency[action]).astype(bool)] = 0

    return next_graph_state, next_knowledge_state, new_knowledge

def sample_and_store_rollouts(graph_state, knowledge_state, adjacency, depth, max_depth, node, select_actions):
    if depth > max_depth:
        return
    
    actions = select_actions(graph_state, knowledge_state)  # Dynamic selection of actions
    if len(actions) == 0:
        return
        
    
    for action in actions:
        next_graph_state, next_knowledge_state, new_knowledge = perform_action(
            graph_state, knowledge_state, action, adjacency
        )  # Replace with your environment dynamics
        
        child_node = Node(next_graph_state, next_knowledge_state, action, new_knowledge, depth)
        node.children.append(child_node)
        if (next_knowledge_state == 0).sum() == 0:
            max_depth = min([child_node.depth])
            # print(f"Path complete: {child_node.depth}")
            return
        
        sample_and_store_rollouts(next_graph_state, next_knowledge_state, adjacency, depth + 1, max_depth, child_node, select_actions)

    return node

def gpt_learn_graph(ngroups, adjacency, max_depth=6):
    initial_graph_state = np.zeros(ngroups)
    initial_knowledge_state = np.zeros((ngroups, ngroups))
    initial_knowledge_state[np.diag_indices(ngroups)] = -1
    root_node = Node(initial_graph_state, initial_knowledge_state, None, None, 0)
    paths = sample_and_store_rollouts(
        initial_graph_state,
        initial_knowledge_state,
        adjacency,
        1,
        max_depth,
        root_node,
        select_actions
    )
    return paths


def learn_the_graph(
    ngroups, 
    nmoves, 
    adjacency, 
    initial_graph_state,
    initial_knowledge_state
    ):
    """
    Thoughts:
    - coudl we say whether in general it is more valuable to gain knowledge now
    by turning on something that will reveal new edges versus gaining knowledge on future turns by turning 
    things on that will make future selections more informative?
    - i guess a future oriented selection is only actually potentially valuable if it doesn't turn off currently on lights
    - if it turns off lights and i already know that's going to happen, it's net loss
    - so its only worth turning on lights where i either dont know whehter it will turn currently on lights off (less valueable)
    or i know it won't turn off currently on lights (more valuable in expectation)
    - i guess the value of turning on a light (A) i already know doesn;t turn off B (currently on) is proportional to the number of lights 
    that i could turn on next round that i currently don't know whether they turn off A (e.g. B and C).
    - more generally, turning something on has these factors that I can always know because i know part of what the next state will be:
        - i know that whatever i select will be on in the next state
        - and there may be some lights that I know will go off for the next state if i select this
        - so you can choose the best action for thinking one step ahead in terms of information gain possible at the next decision
        if you turn each light on, versus the ifnormation you will gain *this*
    """
    pass


def all_valid_paths(DAG):
    # sorts = list(nx.all_topological_sorts(DAG))
    sorts = list(all_topological_sorts(DAG.edges()))
    print(sorts)
    return sorts

def generate_adjacency(DAG, ngroups):
    adjacency = nx.to_numpy_array(DAG, nodelist = range(ngroups)).astype(int)
    n = len(DAG.nodes())
    assert len(adjacency) == n
    print(DAG.edges(), 'degde')
    print(adjacency, 'adj')
    return adjacency.astype(int)

def generate_dag(n):
    print('NGD starting')
    G = nx.gnp_random_graph(n,0.85,directed=True)
    no_bidirectional = remove_symmetric_edges(G.edges())
    print(no_bidirectional, 'nbd')
    DAG = nx.DiGraph(
        [(u,v) for (u,v) in no_bidirectional if u < v] # no cycles (using topological ordering) -- this is too restrictive i think, consider other ways of preventing cycles.
    )
    for i in range(n):
        if i not in DAG.nodes():
            DAG.add_node(i)
    assert nx.is_directed_acyclic_graph(DAG)
    return DAG

def get_data_from_path(path):
    dfs = []
    for node in path:
        df = pivot_array_to_dataframe(node.knowledge_state)
        df['depth'] = node.depth
        df['action'] = node.action
        for i,group in enumerate(node.graph_state):
            df[f'graph_state_{i}'] = int(group)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def pivot_array_to_dataframe(arr):
    # Get row and column indices
    row_indices, col_indices = np.indices(arr.shape)

    # Flatten the indices and values
    indices = np.vstack((row_indices.ravel(), col_indices.ravel())).T
    values = arr.ravel()

    # Create the DataFrame
    df = pd.DataFrame(np.concatenate((indices, values[:, None]), axis=1), columns=['source', 'target', 'edge'])

    return df

def generate_game(ngroups, ncols, nrows):
    dag = generate_dag(ngroups)
    adjacency = generate_adjacency(dag, ngroups)
    state = np.zeros((nrows, ncols)).astype(int)
    rng = np.random.default_rng()
    groups = rng.integers(
        ngroups,
        size=(nrows, ncols)
    ).astype(int)
    return {
        "state": state.tolist(),
        "groups": groups.tolist(),
        'dag': adjacency.tolist()
    }


@click.command()
@click.option('--filepath', '-f', default="")
def run(filepath):
    ngames = 5
    ngroups = 5
    ncols = nrows = 6
    games = [
        generate_game(ngroups, ncols=ncols, nrows=nrows) for k in range(ngames)
    ]
    with open(filepath, 'w') as f:
        json.dump(games[0], f, separators=(',', ':'))
    print(f"Saved to {filepath}")

    # valid_paths = all_valid_paths(dag)
    # root_node = gpt_learn_graph(ngroups=ngroups, adjacency=adjacency, max_depth=8)
    # print("finished calculating root_node")
    # # plot_trees(root_node)
    # # print(root_node.children)
    # path_lengths = find_branch_length_counts(root_node)
    # print(path_lengths)
    # minimum = min(path_lengths.keys())
    # print(minimum)
    # shortest_paths = collect_root_node_with_length(root_node, minimum + 2)
    # print(len(shortest_paths))
    # visualize_paths(shortest_paths)
    # print(get_data_from_path(shortest_paths[0]))

    
    
    # draw_dag(dag)
    # visualize_paths_as_tree(valid_paths, colors_hex)




    


    

if __name__ == '__main__':
    run()