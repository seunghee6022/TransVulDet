import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertConfig

import networkx as nx
from torch.utils.data import Dataset


from src.trainer import CustomTrainer
from src.dataset import CodeDataset, split_dataframe
from src.graph import create_graph_from_json

    
class BertWithHierarchicalClassifier(nn.Module):
    def __init__(self, model_name, embedding_dim, uid_to_dimension, graph):
        super(BertWithHierarchicalClassifier, self).__init__()
        self.model_name = model_name
        self.model = BertModel.from_pretrained(self.model_name)
        
        self.input_dim = self.model.config.hidden_size
        self.embedding_dim = embedding_dim
        self.graph = graph

        # Here, replace BERT's linear classifier with your hierarchical classifier
        self.classifier = HierarchicalClassifier(self.input_dim, self.embedding_dim, self.graph)

        self.uid_to_dimension = uid_to_dimension
        self._force_prediction_targets = True
        self.loss_weights = torch.ones(embedding_dim)

        print(f"{self.input_dim}$$$$$$$$$$$$$$$$$$$$$$INSIDE BertWithHierarchicalClassifier")
    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        print("######################## FORWARD")
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        print(f"outputs :{type(outputs)} {outputs}")
        # I'm assuming you'd like to use the [CLS] token representation as features for your hierarchical classifier
        cls_output = outputs[1]
        print(f"cls_output :{cls_output.shape} {cls_output}")
        logits = self.classifier(cls_output)
        print(f"##################logits: {logits.shape} {logits}")
        if labels is not None:
            print(f"############ labels: {labels.shape} {labels}")
            loss = self.loss(logits, labels)
            return loss, logits
        
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
        # print(embedding.shape, embedding)
        for i, label in enumerate(labels):
            # print(f"[{i}] -- label: {label}")
            if label == 10000:
                embedding[i] = 1.0
            else:
                # print(f"[{i}] - label:{label}, uid_to_dimension[label]:{self.uid_to_dimension[label]}")
                embedding[i, self.uid_to_dimension[label]] = 1.0
                for ancestor in nx.ancestors(self.graph, label):
                    # print("ancestor",ancestor, "uid_to_dimension[ancestor]",self.uid_to_dimension[ancestor])
                    embedding[i, self.uid_to_dimension[ancestor]] = 1.0
         # Convert numpy array to torch tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        return embedding_tensor
    
    def loss(self, logits, one_hot_targets, weight_batch=None, global_step=None):
        '''
        ground_truth should be cwe id values. Given ground_truth is one-hot-encoded so needed to be converted to cwe_id list
        '''
        print("logits: ", logits)
        print("one_hot_targets: ", one_hot_targets)

        targets = self.one_hot_labels_to_cweIDs_labels(one_hot_targets)

        # If weight_batch is not provided, use a tensor of ones with the same shape as feature_batch
        if weight_batch is None:
            weight_batch = torch.ones_like(logits[:, 0])

        loss_mask = np.zeros((len(targets), len(self.uid_to_dimension)))
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32)
        print(f"loss_mask: {loss_mask.shape} {loss_mask}")

        for i, label in enumerate(targets):
            print(f"[{i}] -- label: {label}")
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
        print("embedding",embedding.shape, embedding)
        prediction = logits # forward instead of predict_embedded funtion
        print("prediction", prediction.shape, prediction)

        # Clipping predictions for stability
        clipped_probs = torch.clamp(prediction, 1e-7, 1.0 - 1e-7)
        print("clipped_probs", clipped_probs.shape, clipped_probs)
        
        # Binary cross entropy loss calculation
        the_loss = -(
            embedding * torch.log(clipped_probs) +
            (1.0 - embedding) * torch.log(1.0 - clipped_probs)
        )
        print(f"the_loss: {type(the_loss)} {the_loss}")
        print(f"loss_mask: {type(loss_mask)} {loss_mask}")
        print(f"self.loss_weights: {type(self.loss_weights)} {self.loss_weights}")
        sum_per_batch_element = torch.sum(
            the_loss * loss_mask * self.loss_weights, dim=1
        )
        print("sum_per_batch_element", sum_per_batch_element)
        # This is your L2 regularization term
        l2_penalty = self.classifier.l2_penalty()
        print("l2_penalty", l2_penalty)
        print("torch.mean(sum_per_batch_element * weight_batch)", torch.mean(sum_per_batch_element * weight_batch))
        total_loss = torch.mean(sum_per_batch_element * weight_batch) + l2_penalty
        print("INSIDE HC LOSS FUNCTION -------total_loss: ", total_loss) 
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

    
    # def loss(self, feature_batch, one_hot_ground_truth, weight_batch=None, global_step=None):
    #     '''
    #     ground_truth should be cwe id values. Given ground_truth is one-hot-encoded so needed to be converted to cwe_id list
    #     '''
    #     print("feature_batch: ", feature_batch)
    #     print("one_hot_ground_truth: ", one_hot_ground_truth)

    #     ground_truth = self.one_hot_labels_to_cweIDs_labels(one_hot_ground_truth)

    #     # If weight_batch is not provided, use a tensor of ones with the same shape as feature_batch
    #     if weight_batch is None:
    #         weight_batch = torch.ones_like(feature_batch[:, 0])

    #     loss_mask = np.zeros((len(ground_truth), len(self.uid_to_dimension)))
    #     print(f"loss_mask: {loss_mask.shape} {loss_mask}")

    #     for i, label in enumerate(ground_truth):
    #         print(f"[{i}] -- label: {label}")
    #         # Loss mask
    #         loss_mask[i, self.uid_to_dimension[label]] = 1.0

    #         for ancestor in nx.ancestors(self.graph, label):
    #             loss_mask[i, self.uid_to_dimension[ancestor]] = 1.0
    #             for successor in self.graph.successors(ancestor):
    #                 loss_mask[i, self.uid_to_dimension[successor]] = 1.0
    #                 # This should also cover the node itself, but we do it anyway

    #         if not self._force_prediction_targets:
    #             # Learn direct successors in order to "stop"
    #             # prediction at these nodes.
    #             # If MLNP is active, then this can be ignored.
    #             # Because we never want to predict
    #             # inner nodes, we interpret labels at
    #             # inner nodes as imprecise labels.
    #             for successor in self.graph.successors(label):
    #                 loss_mask[i, self.uid_to_dimension[successor]] = 1.0

    #     embedding = self.embed(ground_truth)
    #     print("embedding",embedding.shape, embedding)
    #     prediction = self.forward(feature_batch) # forward instead of predict_embedded funtion
    #     print("prediction", prediction.shape, prediction)

    #     # Clipping predictions for stability
    #     clipped_probs = torch.clamp(prediction, 1e-7, 1.0 - 1e-7)
    #     print("clipped_probs", clipped_probs.shape, clipped_probs)
        
    #     # Binary cross entropy loss calculation
    #     the_loss = -(
    #         embedding * torch.log(clipped_probs) +
    #         (1.0 - embedding) * torch.log(1.0 - clipped_probs)
    #     )
    #     print("the_loss", the_loss)
    #     sum_per_batch_element = torch.sum(
    #         the_loss * loss_mask * self.loss_weights, dim=1
    #     )
    #     print("sum_per_batch_element", sum_per_batch_element)
    #     # This is your L2 regularization term
    #     l2_penalty = self.l2_regularization_coefficient * torch.sum(self.linear.weight ** 2)
    #     print("l2_penalty", l2_penalty)
    #     print("torch.mean(sum_per_batch_element * weight_batch)", torch.mean(sum_per_batch_element * weight_batch))
    #     total_loss = torch.mean(sum_per_batch_element * weight_batch) + l2_penalty
    #     print("INSIDE HC LOSS FUNCTION -------total_loss: ", total_loss) 
    #     return total_loss
    
def set_uid_to_dimension(graph):
    all_uids = nx.topological_sort(graph)
    print("all_uids\n",all_uids)
    topo_sorted_uids = list(all_uids)
    print("topo_sorted_uids\n",topo_sorted_uids)
    uid_to_dimension = {
            uid: dimension for dimension, uid in enumerate(topo_sorted_uids)
        }
    return uid_to_dimension
   
if __name__ == "__main__":
    print(os.getcwd())
    # # Create graph from JSON
    paths_file = 'data_preprocessing/preprocessed_datasets/graph_all_paths.json'
    with open(paths_file, 'r') as f:
        paths_dict_data = json.load(f)
   
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    '''
    Can be generalized to other model & tokenizer later
    '''
    # Define Tokenizer and Model
    batch_size = 8
    num_labels = graph.number_of_nodes()  # or however many labels you have
    print("num_labels: ", num_labels)
    use_hierarchical_classifier = True
    model_name = 'bert-base-uncased'
    input_dim = 786
    embedding_dim = num_labels
    uid_to_dimension = set_uid_to_dimension(graph)

    if not use_hierarchical_classifier:
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        model = BertForSequenceClassification.from_pretrained(model_name, config=config)
        
    else:
        model = BertWithHierarchicalClassifier(model_name, embedding_dim, uid_to_dimension,graph)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f"use_hierarchical_classifier:{use_hierarchical_classifier} --> model:{model}")

    # Freeze all parameters of the model
    # By setting the requires_grad attribute to False, you can freeze the parameters so they won't be updated during training
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head:
    # To fine-tune only the classifier head, we'll unfreeze its parameters
    print(model.classifier)
    for param in model.classifier.parameters():
        print(param)
        param.requires_grad = True

    # Define Dataset
    # Split the DataFrame dataset into tran/val/test datasets and Tokenize the "code" column of your DataFrame
    df_path = 'data_preprocessing/preprocessed_datasets/MVD_1000.csv'
    max_length = 512
    lr= 1e-4

    train_df, val_df, test_df = split_dataframe(df_path)
    
    train_encodings = tokenizer(list(train_df["code"]), truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    val_encodings = tokenizer(list(val_df["code"]), truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    test_encodings = tokenizer(list(test_df["code"]), truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    train_labels = list(train_df["cwe_id"])
    val_labels = list(val_df["cwe_id"])
    test_labels = list(test_df["cwe_id"])

    print("uid_to_dimension\n",uid_to_dimension)

    train_dataset = CodeDataset(train_encodings, train_labels, uid_to_dimension)
    val_dataset = CodeDataset(val_encodings, val_labels, uid_to_dimension)
    test_dataset = CodeDataset(test_encodings, test_labels, uid_to_dimension)

    print(len(train_labels),len(val_labels), len(test_labels) )
   
    # Define loss function, optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)


    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        logging_dir='./logs',
        output_dir='./outputs',
        evaluation_strategy="steps",
        eval_steps=1,  # Evaluate and log metrics every 500 steps
        logging_steps=1,
        learning_rate=lr,
        remove_unused_columns=False,  # Important for our custom loss function
        disable_tqdm=False,
    )

    trainer = CustomTrainer(
        use_hierarchical_classifier = use_hierarchical_classifier,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # lr_scheduler=scheduler,  # Our custom loss function
    )

    trainer.train()

 