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
from src.graph import create_graph_from_json, set_uid_to_dimension
from src.classifier import BertWithHierarchicalClassifier

   
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

 