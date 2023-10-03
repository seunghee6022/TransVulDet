import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
# import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

import torch
import torch.nn as nn
import torch.nn.functional as F


# Assume x, true_labels are given tensors
# prediction = model(x)
# loss = model.calculate_loss(prediction, true_labels)

# class Classifier(nn.Module):
#     def __init__(self, input_dim, embedding_dim, graph):
#         super(Classifier, self).__init__()
#         self.linear = nn.Linear(input_dim, embedding_dim)
#         self.graph = graph
#         self._force_prediction_targets = True
#         self._l2_regularization_coefficient = 5e-5
#         self.uid_to_dimension = {}
#         self.prediction_target_uids = None
#         self.topo_sorted_uids = None
#         # self.loss_weights = None
#         self.loss_weights = torch.ones(embedding_dim)

#         # Initialize weights and biases to zero
#         nn.init.zeros_(self.linear.weight)
#         nn.init.zeros_(self.linear.bias)

#     def set_uid_to_dimension(self):
#         all_uids = nx.topological_sort(self.graph)
#         print("all_uids\n",all_uids)
#         self.topo_sorted_uids = list(all_uids)
#         print("topo_sorted_uids\n",self.topo_sorted_uids)
#         self.uid_to_dimension = {
#                 uid: dimension for dimension, uid in enumerate(self.topo_sorted_uids)
#             }
    
#         return self.uid_to_dimension
    
#     def forward(self, x):
#         return torch.sigmoid(self.linear(x))
    
#     def predict_class(self, x):
#         return (self.forward(x) > 0.5).float()  # Threshold at 0.5
    
#     def predict_embedded(self, x):
#         return self.forward(x)
    
#     # uid_to_dimension --> dict: {uid: #_dim}
#     def embed(self, labels):
#         embedding = np.zeros((len(labels), len(self.uid_to_dimension)))
#         print(embedding.shape, embedding)
#         for i, label in enumerate(labels):
#             if label == 10000:
#                 embedding[i] = 1.0
#             else:
#                 print(f"[{i}] - label:{label}, uid_to_dimension[label]:{self.uid_to_dimension[label]}")
#                 embedding[i, self.uid_to_dimension[label]] = 1.0
#                 for ancestor in nx.ancestors(self.graph, label):
#                     print("ancestor",ancestor, "uid_to_dimension[ancestor]",self.uid_to_dimension[ancestor])
#                     embedding[i, self.uid_to_dimension[ancestor]] = 1.0
#                     print("embedding",embedding)
#         return embedding
    
#     def loss(self, feature_batch, ground_truth, weight_batch=None, global_step=None):
#         # If weight_batch is not provided, use a tensor of ones with the same shape as feature_batch
#         if weight_batch is None:
#             weight_batch = torch.ones_like(feature_batch[:, 0])

#         loss_mask = np.zeros((len(ground_truth), len(self.uid_to_dimension)))
#         for i, label in enumerate(ground_truth):
#             # Loss mask
#             loss_mask[i, self.uid_to_dimension[label]] = 1.0

#             for ancestor in nx.ancestors(self.graph, label):
#                 loss_mask[i, self.uid_to_dimension[ancestor]] = 1.0
#                 for successor in self.graph.successors(ancestor):
#                     loss_mask[i, self.uid_to_dimension[successor]] = 1.0
#                     # This should also cover the node itself, but we do it anyway

#             if not self._force_prediction_targets:
#                 # Learn direct successors in order to "stop"
#                 # prediction at these nodes.
#                 # If MLNP is active, then this can be ignored.
#                 # Because we never want to predict
#                 # inner nodes, we interpret labels at
#                 # inner nodes as imprecise labels.
#                 for successor in self.graph.successors(label):
#                     loss_mask[i, self.uid_to_dimension[successor]] = 1.0

#         embedding = self.embed(ground_truth)
#         print("embedding",embedding.shape, embedding)
#         prediction = self.predict_embedded(feature_batch)
#         print("prediction", prediction.shape, prediction)

#         # Clipping predictions for stability
#         clipped_probs = torch.clamp(prediction, 1e-7, 1.0 - 1e-7)
#         print("clipped_probs", clipped_probs.shape, clipped_probs)
        
#         # Binary cross entropy loss calculation
#         the_loss = -(
#             embedding * torch.log(clipped_probs) +
#             (1.0 - embedding) * torch.log(1.0 - clipped_probs)
#         )
#         print("the_loss", the_loss)
#         sum_per_batch_element = torch.sum(
#             the_loss * loss_mask * self.loss_weights, dim=1
#         )
#         print("sum_per_batch_element", sum_per_batch_element)
#         # This is your L2 regularization term
#         l2_penalty = self.l2_regularization_coefficient * torch.sum(self.linear.weight ** 2)
#         print("l2_penalty", l2_penalty)
#         print("torch.mean(sum_per_batch_element * weight_batch)", torch.mean(sum_per_batch_element * weight_batch))
#         total_loss = torch.mean(sum_per_batch_element * weight_batch) + l2_penalty
#         print("total_loss", total_loss) 
#         return total_loss
       

def sort_dict(node_levels):
    # Sort dictionary by keys
    sorted_by_key = OrderedDict(sorted(node_levels.items()))

    return sorted_by_key

def set_uid_to_dimension(graph):
    all_uids = nx.topological_sort(graph)
    topo_sorted_uids = list(all_uids)
    uid_to_dimension = {
            uid: dimension for dimension, uid in enumerate(topo_sorted_uids)
        }
    return uid_to_dimension

def find_hierarchy(paths_dict_data, max_level=None):
    hierarchy = {}
    for key, path in paths_dict_data.items():
        items = path.split('-')
        if key in items:
            level = len(items)
            # If max_level is set, only store nodes with hierarchy level less than or equal to max_level
            if max_level is None or level <= max_level:
                hierarchy[int(key)] = level
    return hierarchy


def create_graph_from_json(paths_dict_data, max_depth=None):
    
    G = nx.DiGraph()

    def add_path_to_graph(path):
        nodes = list(map(int, path.split('-')))
        if max_depth:
            max_level = min(max_depth, len(nodes) - 1)
            for i in range(max_level):
                G.add_edge(nodes[i], nodes[i+1])
        else:
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i+1])

    # Add edges from the paths in the JSON data
    for key, paths_list in paths_dict_data.items():
        for path in paths_list:
            add_path_to_graph(path)
            
    return G


def show_tree_graph(graph):
 
    plt.figure(figsize=(30, 8))
    plt.tight_layout(pad=1.0)
    # plt.ylim(0, 5)  # Adjust as needed

    # Assuming pos has your node positions
    scale_factor_x = 1.2  # adjust as needed
    scale_factor_y = 0.8
    pos = graphviz_layout(graph, prog='dot')  # Use the Graphviz layout
    # print("pos\n",pos)
    pos = {node: (x * scale_factor_x, y * scale_factor_y) for node, (x, y) in pos.items()}
    # print("pos\n",pos)
    nx.draw(G, pos=pos, with_labels=True, node_size=100, node_color="skyblue",font_size=4, width=1)
    # Set the title for the figure
    plt.title("Directed Acyclic Graph for CWE Hierarchy")
    plt.show()



# if __name__ == "__main__":
#     # Create graph from JSON
#     paths_file = 'graph_all_paths.json'
#     with open(paths_file, 'r') as f:
#         paths_dict_data = json.load(f)
    
#     mvd_df = pd.read_csv('MVD_6.csv', index_col=0)
#     print(mvd_df)
    
#     labels = list(mvd_df['cwe_id'])
#     print(type(labels), labels)

#     max_depth = None
#     G = create_graph_from_json(paths_dict_data, max_depth)
#     # Draw the graph in a tree style
#     # show_tree_graph(G)

#     # Example of using the classifier
#     input_dim = 10
#     embedding_dim = 5 #232 num of total nodes (not target nodes)

#     uid_to_dimension = set_uid_to_dimension(G)
#     print("uid_to_dimension",len(uid_to_dimension), uid_to_dimension)

#     prediction_target_uids = [int(key) for key in paths_dict_data.keys()]
#     print("prediction_target_uids",len(prediction_target_uids), prediction_target_uids)

#     # feature_batch = []
#     # ground_truth = []
#     # loss = classifier.loss(feature_batch, ground_truth)


    