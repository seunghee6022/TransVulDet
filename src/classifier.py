import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertConfig
import networkx as nx

class BertWithHierarchicalClassifier(nn.Module):
    def __init__(self, model_name, embedding_dim, uid_to_dimension, graph):
        super(BertWithHierarchicalClassifier, self).__init__()
        self.model_name = model_name
        self.model = BertModel.from_pretrained(self.model_name)
        
        self.input_dim = self.model.config.hidden_size
        self.embedding_dim = embedding_dim
        self.graph = graph

        # Here, replace BERT's linear classifier with hierarchical classifier
        self.classifier = HierarchicalClassifier(self.input_dim, self.embedding_dim, self.graph)

        self.uid_to_dimension = uid_to_dimension
        self._force_prediction_targets = True
        self.loss_weights = torch.ones(embedding_dim)

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        print("INSIDE BertWithHierarchicalClassifier Forward")
    
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # use the [CLS] token representation as features for hierarchical classifier
        cls_output = outputs[1]
        logits = self.classifier(cls_output)
        print("logits---",logits.shape)
        if labels is not None:
            loss = self.loss(logits, labels)
            print("labels is not None---calculated loss:",loss)
            return loss, logits
        else:
            return logits
    
    def one_hot_labels_to_cweIDs_labels(self, ground_truth):
            '''
            ground_truth is one-hot-encoded
            uid: cwe id
            '''
            # Reverse the dictionary
            dim_to_uid = {v: k for k, v in self.uid_to_dimension.items()}

            # Get the indices where value is 1 in each row of ground_truth and map to keys
            # indices = [row.nonzero().item() for row in ground_truth]
            indices = torch.argmax(ground_truth, dim=1).tolist()
            uid_labels = [dim_to_uid[idx] for idx in indices]
           
            return uid_labels
    
    def embed(self, labels):
        embedding = np.zeros((len(labels), len(self.uid_to_dimension)))
        
        for i, label in enumerate(labels):
            if label == 10000:
                embedding[i] = 1.0
            else:
                embedding[i, self.uid_to_dimension[label]] = 1.0
                for ancestor in nx.ancestors(self.graph, label):
                    embedding[i, self.uid_to_dimension[ancestor]] = 1.0
         # Convert numpy array to torch tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        return embedding_tensor
    
    def loss(self, logits, one_hot_targets, weight_batch=None, global_step=None):
        '''
        ground_truth should be cwe id values. Given ground_truth is one-hot-encoded so needed to be converted to cwe_id list
        '''
        
        targets = self.one_hot_labels_to_cweIDs_labels(one_hot_targets)

        # If weight_batch is not provided, use a tensor of ones with the same shape as logits
        if weight_batch is None:
            weight_batch = torch.ones_like(logits[:, 0])

        loss_mask = np.zeros((len(targets), len(self.uid_to_dimension)))
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32)
        
        for i, label in enumerate(targets):
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

        embedding = self.embed(targets)
        prediction = logits # forward instead of predict_embedded funtion
        
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
        
        # This is L2 regularization term
        l2_penalty = self.classifier.l2_penalty()
        total_loss = torch.mean(sum_per_batch_element * weight_batch) + l2_penalty
        print("INSIDE BertWithHierarchicalClassifier ---- INSIDE loss function------total_loss--------",total_loss)
        return total_loss

class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim=786, embedding_dim=None, graph=None):
        super(HierarchicalClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation layer
        self._l2_regularization_coefficient = 5e-5
       
        # Initialize weights and biases to zero
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        x = self.linear(x)  # Linear transformation
        x = self.sigmoid(x)  # Apply sigmoid activation function
        return x
    
    def l2_penalty(self):
        return self._l2_regularization_coefficient * torch.sum(self.linear.weight ** 2)