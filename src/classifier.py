import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from transformers.modeling_utils import PreTrainedModel ,PretrainedConfig
from transformers import (
    AutoTokenizer,AutoModel,AutoModelForSequenceClassification,RobertaConfig
)

def get_model_and_tokenizer(args, prediction_target_uids,graph):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = RobertaConfig()
    if args.use_hierarchical_classifier or args.use_bilstm:
        model = TransformerWithHierarchicalClassifier(
            args.model_name, prediction_target_uids, graph, args.use_bilstm, args.use_hierarchical_classifier, args.loss_weight, embedding_dim=768) #config=config
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(prediction_target_uids))
    return model, tokenizer

class TransformerWithHierarchicalClassifier(nn.Module):
    def __init__(self, model_name, prediction_target_uids, graph, use_bilstm, use_hierarchical_classifier, _weighting='equalize',embedding_dim=768):
        super(TransformerWithHierarchicalClassifier, self).__init__()
        self.use_hierarchical_classifier = use_hierarchical_classifier
        self.embedding_dim = embedding_dim
        self.graph = graph
        self._force_prediction_targets = True
        self.prediction_target_uids = prediction_target_uids
         # Here, replace BERT's linear classifier with hierarchical classifier
        self.topo_sorted_uids = None
        self.uid_to_dimension = None
        self.set_uid_to_dimension_and_topo_sorted_uids() # set the uid_to_dimension and topo_sorted_uids
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_bilstm = use_bilstm
        self.bidirectional = True
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.lstm_hidden_dim = 256
        self.input_dim= self.lstm_hidden_dim*2 if self.use_bilstm  else self.embedding_dim

        if self.use_bilstm:
            
            self.bilstm =  torch.nn.LSTM(input_size=self.embedding_dim, 
                               hidden_size=self.lstm_hidden_dim,
                               num_layers=2,
                               bidirectional=self.bidirectional,
                               batch_first=True) #(batch, seq, feature)
            if self.use_hierarchical_classifier:
                self.classifier = HierarchicalClassifier(self.input_dim, len(self.uid_to_dimension), self.graph)
            else:
                self.fc = nn.Linear(self.input_dim, len(self.prediction_target_uids))
        
        
        self.classifier = HierarchicalClassifier(self.input_dim, len(self.uid_to_dimension), self.graph)
        self._weighting = _weighting
        self.loss_weights = np.ones(len(self.uid_to_dimension))
        self.get_loss_weight()

    def set_uid_to_dimension_and_topo_sorted_uids(self):
        all_uids = nx.topological_sort(self.graph)
        self.topo_sorted_uids = list(all_uids)
        self.uid_to_dimension = {
                uid: dimension for dimension, uid in enumerate(self.topo_sorted_uids)
            }
    def get_loss_weight(self):
         # (1) Calculate "natural" weights by assuming uniform distribution
        # over observed concepts
        occurences = {uid: 0 for uid in self.topo_sorted_uids}
        for uid in self.prediction_target_uids:
            affected_uids = {uid}
            affected_uids |= nx.ancestors(self.graph, uid)
            for affected_uid in list(affected_uids):
                affected_uids |= set(self.graph.successors(affected_uid))

            for affected_uid in affected_uids:
                occurences[affected_uid] += 1

        occurrence_vector = np.array([occurences[uid] for uid in self.uid_to_dimension])
    
        # (2) Calculate weight vector
        if self._weighting == "default":
            self.loss_weights = np.ones(len(self.uid_to_dimension))
        
        elif self._weighting == "equalize":
            try:
                self.loss_weights = (
                    np.ones(len(self.uid_to_dimension)) / occurrence_vector
                )
            except ZeroDivisionError as err:
                self.log_fatal("Division by zero in equalize loss weighting strategy.")
                raise err

        elif self._weighting == "descendants":
            try:
                # Start with an equal weighting
                self.loss_weights = (
                    np.ones(len(self.uid_to_dimension)) / occurrence_vector
                )

                for i, uid in enumerate(self.uid_to_dimension):
                    self.loss_weights[i] *= (
                        len(nx.descendants(self.graph, uid)) + 1.0
                       
                    )  # Add one for the node itself.
                
            except ZeroDivisionError as err:
                self.log_fatal(
                    "Division by zero in descendants loss weighting strategy."
                )
                raise err
        elif self._weighting == "reachable_leaf_nodes":
            try:
                # Start with an equal weighting
                self.loss_weights = (
                    np.ones(len(self.uid_to_dimension)) / occurrence_vector
                )

                for i, uid in enumerate(self.uid_to_dimension):
                    descendants = set(nx.descendants(self.graph, uid)) | {uid}
                    reachable_leaf_nodes = descendants.intersection(
                        self.prediction_target_uids
                    )
                    
                    self.loss_weights[i] *= len(reachable_leaf_nodes)

                    # Test if any leaf nodes are reachable
                    if len(reachable_leaf_nodes) == 0:
                        raise ValueError(
                            f"In this hierarchy, the node {uid} cannot reach "
                            "any leaf nodes!"
                        )

            except ZeroDivisionError as err:
                self.log_fatal(
                    "Division by zero in reachable_leaf_nodes loss weighting strategy."
                )
                raise err
        self.loss_weights = torch.tensor(self.loss_weights, dtype=torch.float32)
       
    def forward(self, input_ids, attention_mask=None,  labels=None, **kwargs):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        labels=None
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
        if self.use_bilstm:
            lstm_out, _ = self.bilstm(last_hidden_state)
            lstm_out = lstm_out[:, -1, :]
            if self.use_hierarchical_classifier:
                logits = self.classifier(lstm_out)
                
            else:
                logits = self.fc(lstm_out)

        else:
            cls_output = last_hidden_state[:, 0, :]
            logits = self.classifier(cls_output)
        
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, logits
        else:
            return logits
    
    
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
    
    
    def loss(self, logits, targets, weight_batch=None, global_step=None):
        '''
        ground_truth should be cwe id values. Given ground_truth is one-hot-encoded so needed to be converted to cwe_id list
        '''
        targets = targets.tolist()
       
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
        # print("loss_mask", loss_mask)

        embedding = self.embed(targets)
        prediction = logits # batch*size, len(uid_to_dimension) : 32*32
      
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
        return total_loss
    
    def _deembed_single(self, embedded_label):
        conditional_probabilities = {
            uid: embedded_label[i] for uid, i in self.uid_to_dimension.items()
        }
        
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

        # Stage 2 calculates the joint probability of the synset and "no children"
        joint_probabilities = {}
        for uid in reversed(self.topo_sorted_uids):
            joint_probability = unconditional_probabilities[uid]
            no_child_probability = 1.0
            for child in self.graph.successors(uid):
                no_child_probability *= 1.0 - unconditional_probabilities[child]

            joint_probabilities[uid] = joint_probability * no_child_probability
        
        tuples = joint_probabilities.items()
        sorted_tuples = list(sorted(tuples, key=lambda tup: tup[1], reverse=True))
        
        # If requested, only output scores for the forced prediction targets
        if self._force_prediction_targets:
            for i, (uid, p) in enumerate(sorted_tuples):
                if uid not in self.prediction_target_uids:
                    sorted_tuples[i] = (uid, 0.0)
            
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
    
    def dist_to_cwe_ids(self, pred_dist):
        max_cwe_id_list = []
        for sorted_tuples in pred_dist:
            max_cwe_id = max(sorted_tuples, key=lambda x: x[1])[0]
            max_cwe_id_list.append(max_cwe_id)
        return max_cwe_id_list
    
    def dimension_to_cwe_id(self, idx_labels):
        dimension_to_uid = {v:k for k, v in self.uid_to_dimension.items()}
        cwe_id_list = [dimension_to_uid[idx] for idx in idx_labels]
        return cwe_id_list

class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim=768, output_dim=None, graph=None):
        super(HierarchicalClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation layer
        self._l2_regularization_coefficient = 5e-5
       
        # Initialize weights and biases to zero
        # nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.linear.weight) 
        nn.init.zeros_(self.linear.bias) 
        
    def forward(self, x):
        x = self.linear(x)  # Linear transformation
        x = self.sigmoid(x)  # Apply sigmoid activation function
        return x
    
    def l2_penalty(self):
        return self._l2_regularization_coefficient * torch.sum(self.linear.weight ** 2)
