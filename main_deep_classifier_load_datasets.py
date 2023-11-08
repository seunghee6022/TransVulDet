import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, TrainingArguments, DataCollatorWithPadding
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertConfig
from transformers.trainer_callback import EarlyStoppingCallback

import matplotlib.pyplot as plt

from src.trainer import CustomTrainer
from src.dataset import CodeDataset, split_dataframe, make_repeat_dataset
from src.graph import create_graph_from_json, set_uid_to_dimension
from src.classifier import BertWithHierarchicalClassifier
from callback import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import optuna
from datasets import load_dataset


if __name__ == "__main__":
    # Create graph from JSON
    paths_file = 'data_preprocessing/preprocessed_datasets/debug_datasets/graph_all_paths.json'
    with open(paths_file, 'r') as f:
        paths_dict_data = json.load(f)
   
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    # Define Tokenizer and Model
    batch_size = 32
    num_labels = graph.number_of_nodes() 
    print("num_labels: ", num_labels)
    use_hierarchical_classifier = True
    model_name = 'bert-base-uncased'
    embedding_dim = num_labels
    uid_to_dimension = set_uid_to_dimension(graph)
    lr= 1e-2
    num_epoch = 3

    # Check if a GPU is available and use it, otherwise, use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_hierarchical_classifier:
        model = BertWithHierarchicalClassifier(model_name, prediction_target_uids, graph, embedding_dim)
    else:
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        model = BertForSequenceClassification.from_pretrained(model_name, config=config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"use_hierarchical_classifier:{use_hierarchical_classifier} --> \nmodel:{model}")

    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head: to fine-tune only the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    # Define Dataset
    max_length = 512

    def one_hot_encode(labels):
        print("labels:",labels)
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
    df_path = 'datasets_'
    data_files = {
    'train': f'{df_path}/train.csv',
    'validation': f'{df_path}/val.csv',
    'test': f'{df_path}/test.csv'
    }
    dataset = load_dataset('csv', data_files=data_files)
    # Set the transform function for on-the-fly tokenization
    dataset = dataset.with_transform(encode)
    print(dataset)

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    def compute_metrics(p):
        print("%%%%%%%%%%%%%%%%INSIDE COMPUTE METRICS")
        print("p:",p)
        predictions, labels = p.predictions, p.label_ids
        # print(f"prediction:{predictions.shape} {type(predictions)}\nlabels:{labels.shape}{type(labels)}")
        print(f"prediction:{predictions.shape}\nlabels:{labels}")
        pred_dist = model.deembed_dist(predictions) # get probabilities of each nodes
        print(f"pred_dist: \n{len(pred_dist)}")
        pred_labels = model.dist_to_cwe_ids(pred_dist)
        predictions = pred_labels
        print(f"pred_labels:{len(pred_labels)}")
        # idx_labels = np.argmax(labels, axis=-1)
        # labels = model.dimension_to_cwe_id(idx_labels)
        print(f"labels: {len(labels)}")
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
    optimizer = AdamW(model.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    callbacks = [EarlyStoppingCallback(patience=5, threshold=0)]
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epoch,
        weight_decay=0.01,
        logging_dir='./logs',
        output_dir='./outputs',
        evaluation_strategy="steps",
        eval_steps=200,  
        logging_steps=100,
        learning_rate=lr,
        remove_unused_columns=False,  # Important for our custom loss function
        disable_tqdm=False,
        load_best_model_at_end = True,
        metric_for_best_model = "balanced_accuracy",
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
        callbacks=callbacks,
       
    )

    trainer.train()
    '''
    # Define the directory for saving figures
    figure_dir = os.path.join('figures', f'lr{lr}_bs{batch_size}_epoch{num_epoch}_{dataset_name}')

    # Create the directory if it doesn't exist
    os.makedirs(figure_dir, exist_ok=True)
    # Evaluate the model on the test dataset
    eval_results = trainer.evaluate(test_dataset)

    # Print the evaluation results
    print("Evaluation results:", eval_results)
    print("trainer.state",trainer.state)
    # Access the loss and metrics from the trainer's history
    print("trainer.state.log_history",trainer.state.log_history)
    train_losses = trainer.state.log_history["eval_loss"]
    val_accs = trainer.state.log_history["eval_accuracy"]
    val_f1_scores = trainer.state.log_history["eval_f1_score"]

    # Plot and save the loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(figure_dir, 'loss_curve.png'))
    plt.close()

    # Plot and save the accuracy curve
    plt.figure(figsize=(12, 6))
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(figure_dir, 'accuracy_curve.png'))
    plt.close()

    # Plot and save the F1 score curve
    plt.figure(figsize=(12, 6))
    plt.plot(val_f1_scores, label="Validation F1 Score")
    plt.xlabel("Step")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(os.path.join(figure_dir, 'f1_score_curve.png'))
    plt.close()

    # Print final evaluation metrics
    final_val_acc = val_accs[-1]
    final_val_f1 = val_f1_scores[-1]
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Validation F1 Score: {final_val_f1:.4f}")
    '''