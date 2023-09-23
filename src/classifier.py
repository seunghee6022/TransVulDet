import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertConfig

import networkx as nx



class BertWithHierarchicalClassifier(nn.Module):
    def __init__(self, config: BertConfig, input_dim, embedding_dim, graph):
        super(BertWithHierarchicalClassifier, self).__init__()
        self.model = BertModel(config)
        self.graph = graph
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Here, replace BERT's linear classifier with your hierarchical classifier
        self.classifier = HierarchicalClassifier(self.input_dim, self.embedding_dim, self.graph)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        # I'm assuming you'd like to use the [CLS] token representation as features for your hierarchical classifier
        cls_output = outputs[1]
        logits = self.classifier(cls_output)
        
        return logits

class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, graph):
        super(HierarchicalClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.graph = graph
        self._force_prediction_targets = True
        self._l2_regularization_coefficient = 5e-5
        self.uid_to_dimension = {}
        self.prediction_target_uids = None
        self.topo_sorted_uids = None
        self.loss_weights = torch.ones(embedding_dim)

        # Initialize weights and biases to zero
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def set_uid_to_dimension(self):
        all_uids = nx.topological_sort(self.graph)
        self.topo_sorted_uids = list(all_uids)
        self.uid_to_dimension = {
                uid: dimension for dimension, uid in enumerate(self.topo_sorted_uids)
            }
    
        return self.uid_to_dimension
    
    # predict_embedded funtion
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    # uid_to_dimension --> dict: {uid: #_dim}
    def embed(self, labels):
        embedding = np.zeros((len(labels), len(self.uid_to_dimension)))
        for i, label in enumerate(labels):
            if label == 10000:
                embedding[i] = 1.0
            else:
                embedding[i, self.uid_to_dimension[label]] = 1.0
                for ancestor in nx.ancestors(self.graph, label):
                    embedding[i, self.uid_to_dimension[ancestor]] = 1.0
                   
        return embedding
    
    def loss(self, feature_batch, ground_truth, weight_batch=None, global_step=None):
        # If weight_batch is not provided, use a tensor of ones with the same shape as feature_batch
        if weight_batch is None:
            weight_batch = torch.ones_like(feature_batch[:, 0])

        loss_mask = np.zeros((len(ground_truth), len(self.uid_to_dimension)))
        for i, label in enumerate(ground_truth):
            # Loss mask
            loss_mask[i, self.uid_to_dimension[label]] = 1.0

            for ancestor in nx.ancestors(self.graph, label):
                loss_mask[i, self.uid_to_dimension[ancestor]] = 1.0
                for successor in self.graph.successors(ancestor):
                    loss_mask[i, self.uid_to_dimension[successor]] = 1.0
                    # This should also cover the node itself, but we do it anyway

            if not self._force_prediction_targets:
                # Learn direct successors in order to "stop"
                # prediction at these nodes.
                # If MLNP is active, then this can be ignored.
                # Because we never want to predict
                # inner nodes, we interpret labels at
                # inner nodes as imprecise labels.
                for successor in self.graph.successors(label):
                    loss_mask[i, self.uid_to_dimension[successor]] = 1.0

        embedding = self.embed(ground_truth)
        prediction = self.forward(feature_batch) # forward instead of predict_embedded funtion
        
        # Clipping predictions for stability
        clipped_probs = torch.clamp(prediction, 1e-7, 1.0 - 1e-7)
        
        # Binary cross entropy loss calculation
        the_loss = -(
            embedding * torch.log(clipped_probs) +
            (1.0 - embedding) * torch.log(1.0 - clipped_probs)
        )
        sum_per_batch_element = torch.sum(
            the_loss * loss_mask * self.loss_weights, dim=1
        )
        # This is your L2 regularization term
        l2_penalty = self.l2_regularization_coefficient * torch.sum(self.linear.weight ** 2)
        total_loss = torch.mean(sum_per_batch_element * weight_batch) + l2_penalty
        
        return total_loss