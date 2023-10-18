import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, TrainingArguments, DataCollatorWithPadding
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertConfig
# from transformers.trainer_callback import EarlyStoppingCallback

import matplotlib.pyplot as plt

from src.trainer import CustomTrainer
from src.dataset import CodeDataset, split_dataframe
from src.graph import create_graph_from_json, set_uid_to_dimension
from src.classifier import BertWithHierarchicalClassifier
from src.callback import EarlyStoppingCallback, WandbCallback, OptunaPruningCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import optuna
from datasets import load_dataset

import argparse
import wandb

# Objective function for Optuna
def objective(trial, args):
    
    # Access command line arguments using args.<argument_name>
    data_dir = args.data_dir
    node_paths_dir = args.node_paths_dir
    model_name = args.model_name
    num_train_epochs = args.num_train_epochs
    max_length = args.max_length
    use_hierarchical_classifier = args.use_hierarchical_classifier
    # use_full_datasets = args.use_full_datasets
 
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-2)
    # per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 1, 32, log=True)
    per_device_train_batch_size = 1
    loss_weight_method = trial.suggest_categorical('loss_weight_method', ['default', 'eqaulize', 'descendants','reachable_leaf_nodes'])
    
    # Create graph from JSON
    with open(node_paths_dir, 'r') as f:
        paths_dict_data = json.load(f)
   
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    # Define Tokenizer and Model
    num_labels = graph.number_of_nodes() 
    print("num_labels: ", num_labels)
    embedding_dim = num_labels
    uid_to_dimension = set_uid_to_dimension(graph)
   
    # Check if a GPU is available and use it, otherwise, use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_hierarchical_classifier:
        model = BertWithHierarchicalClassifier(model_name, prediction_target_uids, graph, loss_weight_method, embedding_dim)
    else:
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        model = BertForSequenceClassification.from_pretrained(model_name, config=config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"use_hierarchical_classifier:{use_hierarchical_classifier} --> \nmodel:{model}")
    wandb.watch(model)

    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head: to fine-tune only the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    # Define Dataset

    def one_hot_encode(labels):
        one_hot_encoded = []
        for label in labels:
            one_hot = [0] * num_labels
            one_hot[uid_to_dimension[label]] = 1
            one_hot_encoded.append(one_hot) 
        print("one_hot_encoded",one_hot_encoded) 
        return torch.tensor(one_hot_encoded)
    
    # Function to tokenize on the fly
    def encode(example):
        tokenized_inputs = tokenizer(example['code'], truncation=True, padding=True, max_length=max_length,return_tensors="pt").to(device)
        # tokenized_inputs['labels'] = one_hot_encode(example['cwe_id'])
        tokenized_inputs['labels'] = example['cwe_id']
        return tokenized_inputs

    # Load dataset and make huggingface datasts
  
    data_files = {
    'train': f'{data_dir}/train_data.csv',
    'validation': f'{data_dir}/val_data.csv',
    'test': f'{data_dir}/test_data.csv'
    }
    
    dataset = load_dataset('csv', data_files=data_files)
    # Set the transform function for on-the-fly tokenization
    # dataset = dataset.with_transform(encode) #set_transform
    dataset.set_transform(encode)
    print(dataset)

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    def compute_metrics(p):
        print("%%%%%%%%%%%%%%%%INSIDE COMPUTE METRICS")
        predictions, labels = p.predictions, p.label_ids
        # print(f"prediction:{predictions.shape} {type(predictions)}\nlabels:{labels.shape}{type(labels)}")
        # print(f"prediction:{predictions}\nlabels:{labels}")
        pred_dist = model.deembed_dist(predictions) # get probabilities of each nodes
        # print(f"pred_dist: \n{pred_dist}")
        pred_labels = model.dist_to_cwe_ids(pred_dist)
        predictions = pred_labels
        print(f"pred_labels:{pred_labels}")
        # idx_labels = np.argmax(labels, axis=-1)
        # labels = model.dimension_to_cwe_id(idx_labels)
        print(f"labels: {labels}")
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {
            "balanced_accuracy":balanced_acc,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Define loss function, optimizer and scheduler
    # optimizer = AdamW(model.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir='./logs',
        output_dir='./outputs',
        evaluation_strategy="steps",
        eval_steps=5000,  
        logging_steps=1000,
        learning_rate=lr,
        remove_unused_columns=False,  # Important for our custom loss function
        disable_tqdm=False,
        load_best_model_at_end = True,
        metric_for_best_model = "f1",
        greater_is_better = True,
    )

    trainer = CustomTrainer(
        use_hierarchical_classifier = use_hierarchical_classifier,
        uid_to_dimension = uid_to_dimension,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[OptunaPruningCallback(trial=trial, max_steps=30000),WandbCallback],  
        # callbacks=[EarlyStoppingCallback(patience=5, threshold=0),WandbCallback],
    )

    # Train and evaluate the model
    trainer.train()
    metrics = trainer.evaluate()
    # Log metrics to wandb
    wandb.log(metrics)
    print("metrics:",metrics)

    # Return the metric we want to optimize (e.g., negative of accuracy for maximization)
    # return metrics["eval_balanced_accuracy"]
    return metrics["f1"]
    

if __name__ == "__main__":
    # Initialize a new run
    wandb.init(project="TransVulDet", name="HPO_with_small_datasets")

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Hyperparameter optimization using Optuna")

    # Add arguments
    parser.add_argument('--data-dir', type=str, default='datasets_', help='Path to the dataset directory')
    parser.add_argument('--node-paths-dir', type=str, default='data_preprocessing/preprocessed_datasets/debug_datasets/graph_all_paths.json', help='Path to the dataset directory')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased', help='Name of the model to use')
    parser.add_argument('--num-trials', type=int, default=5, help='Number of trials for Optuna')
    parser.add_argument('--use-hierarchical-classifier', action='store_true', help='Flag for hierarchical classification') #--use-hierarchical-classifier --> true
    # parser.add_argument('--use-full-datasets', action='store_true', help='Flag for using full datasets(combined 3 datasets)') #absent --> false
    parser.add_argument('--num-train-epochs', type=int, default=5, help='Number of epoch for training')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum length for token number')

    # Parse the command line arguments
    args = parser.parse_args()
    print("MAIN - args",args)

    print(os.getcwd())

    # Access command line arguments using args.<argument_name>
    n_trials = args.num_trials

    print(os.getcwd())
    # Initialize Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource="auto", reduction_factor=3
    ),
        
        )
    study.optimize(lambda trial: objective(trial, args), n_trials=n_trials)
    
    # Print results
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best accuracy: {study.best_value}")

# python main.py \
# --model-name "bert-base-uncased" \
# --num-trials 1 \
# --use-hierarchical-classifier 

# deepspeed main_ds.py \
# --model-name "bert-base-uncased" \
# --num-trials 1 \
# --use-hierarchical-classifier 

#      python -c 'from transformers import AutoModel; \
# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
# model = AutoModel.from_pretrained("bert-base-uncased"); \
# estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'