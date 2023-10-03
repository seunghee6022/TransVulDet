import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertConfig
import networkx as nx

class BertWithHierarchicalClassifier(nn.Module):
    def __init__(self, model_name, prediction_target_uids, graph, embedding_dim=768):
        super(BertWithHierarchicalClassifier, self).__init__()
        self.model_name = model_name
        self.model = BertModel.from_pretrained(self.model_name)
        
        self.input_dim = self.model.config.hidden_size
        self.embedding_dim = embedding_dim
        self.graph = graph
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Here, replace BERT's linear classifier with hierarchical classifier
        self.classifier = HierarchicalClassifier(self.input_dim, self.embedding_dim, self.graph)

        self._force_prediction_targets = True
        self.prediction_target_uids = prediction_target_uids
        self.loss_weights = torch.ones(embedding_dim)
        self.topo_sorted_uids = None
        self.uid_to_dimension = None
        self.set_uid_to_dimension_and_topo_sorted_uids() # set the uid_to_dimension and topo_sorted_uids
        print(f"self.uid_to_dimension:{self.uid_to_dimension}\nself.topo_sorted_uids:{self.topo_sorted_uids}")
    def set_uid_to_dimension_and_topo_sorted_uids(self):
        all_uids = nx.topological_sort(self.graph)
        self.topo_sorted_uids = list(all_uids)
        self.uid_to_dimension = {
                uid: dimension for dimension, uid in enumerate(self.topo_sorted_uids)
            }
        
    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        # print("INSIDE BertWithHierarchicalClassifier Forward")
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # use the [CLS] token representation as features for hierarchical classifier
        # print("outputs", outputs)
        cls_output = outputs[1] # 1: pooler_output --> shape (batch_size:8, hidden_size:768), 0: hidden_states
        logits = self.classifier(cls_output)
        # print("logits---",logits.shape)
        if labels is not None:
            loss = self.loss(logits, labels)
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
        # print(f"embedding {embedding.shape} {embedding}")s
        prediction = logits # forward instead of predict_embedded funtion
        # print(f"prediction {prediction.shape} {prediction}")
        
        # Clipping predictions for stability
        clipped_probs = torch.clamp(prediction, 1e-7, 1.0 - 1e-7)

        # debug
        embedding = embedding.to(self.device)
        clipped_probs = clipped_probs.to(self.device)
        loss_mask = loss_mask.to(self.device)
        self.loss_weights = self.loss_weights.to(self.device)
       
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
        # print("INSIDE BertWithHierarchicalClassifier ---- INSIDE loss function------total_loss--------",total_loss)
        return total_loss
    
    def _deembed_single(self, embedded_label):
        conditional_probabilities = {
            uid: embedded_label[i] for uid, i in self.uid_to_dimension.items()
        }
        # print(f"conditional_probabilities:\n {conditional_probabilities}")
     
        # Stage 1 calculates the unconditional probabilities
        unconditional_probabilities = {}

        for uid in self.topo_sorted_uids:
            unconditional_probability = conditional_probabilities[uid]

            no_parent_probability = 1.0
            has_parents = False
            for parent in self.graph.predecessors(uid):
                has_parents = True
                no_parent_probability *= 1.0 - unconditional_probabilities[parent]

            if has_parents:
                unconditional_probability *= 1.0 - no_parent_probability

            unconditional_probabilities[uid] = unconditional_probability
        # print(f"unconditional_probabilities:\n {unconditional_probabilities}")

        # Stage 2 calculates the joint probability of the synset and "no children"
        joint_probabilities = {}
        for uid in reversed(self.topo_sorted_uids):
            joint_probability = unconditional_probabilities[uid]
            no_child_probability = 1.0
            for child in self.graph.successors(uid):
                no_child_probability *= 1.0 - unconditional_probabilities[child]

            joint_probabilities[uid] = joint_probability * no_child_probability
        # print(f"joint_probabilities:\n {joint_probabilities}")
        tuples = joint_probabilities.items()
        sorted_tuples = list(sorted(tuples, key=lambda tup: tup[1], reverse=True))
        # print(f"sorted_tuples:\n {sorted_tuples}")

        # If requested, only output scores for the forced prediction targets
        if self._force_prediction_targets:
            for i, (uid, p) in enumerate(sorted_tuples):
                if uid not in self.prediction_target_uids:
                    sorted_tuples[i] = (uid, 0.0)
            # print(f"sorted_tuples:\n{sorted_tuples}")
            total_scores = sum([p for uid, p in sorted_tuples])
            if total_scores > 0:
                sorted_tuples = [
                    (uid, p / total_scores) for uid, p in sorted_tuples
                ]

            return list(sorted_tuples)
        
    def deembed_dist(self, embedded_labels):
        return [
            self._deembed_single(embedded_label) for embedded_label in embedded_labels
        ]
    
    def dist_to_labels(self, pred_dist):
        max_idx_list = []
        for sorted_tuples in pred_dist:
            max_cwe_id = max(sorted_tuples, key=lambda x: x[1])[0]
            # print(f"max_cwe_id:{max_cwe_id}")
            # print(f"self.uid_to_dimension[max_cwe_id]:{self.uid_to_dimension[max_cwe_id]}")
            max_index = self.uid_to_dimension[max_cwe_id]
            max_idx_list.append(max_index)
        return max_idx_list




class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim=786, embedding_dim=None, graph=None):
        super(HierarchicalClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation layer
        self._l2_regularization_coefficient = 5e-5
       
        # Initialize weights and biases to zero
        nn.init.zeros_(self.linear.weight) # initialize to 0 --> ask..? with not 0?
        nn.init.zeros_(self.linear.bias) 
        
    def forward(self, x):
        x = self.linear(x)  # Linear transformation
        x = self.sigmoid(x)  # Apply sigmoid activation function
        return x
    
    def l2_penalty(self):
        return self._l2_regularization_coefficient * torch.sum(self.linear.weight ** 2)