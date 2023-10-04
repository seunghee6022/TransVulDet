import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, DataCollatorWithPadding
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertConfig
from transformers.trainer_callback import EarlyStoppingCallback

import matplotlib.pyplot as plt

from src.trainer import CustomTrainer
from src.dataset import CodeDataset, split_dataframe
from src.graph import create_graph_from_json, set_uid_to_dimension
from src.classifier import BertWithHierarchicalClassifier
from src.early_stopping import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1])
    }

def model_init(trial):
    
    paths_file = 'data_preprocessing/preprocessed_datasets/debug_datasets/graph_all_paths.json'
    with open(paths_file, 'r') as f:
        paths_dict_data = json.load(f)
   
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
    graph = create_graph_from_json(paths_dict_data, max_depth=None)
    num_labels = graph.number_of_nodes()
    # Setup custom model
    model = BertWithHierarchicalClassifier(
        model_name="bert-base-uncased",
        prediction_target_uids=prediction_target_uids,
        graph=graph,
        embedding_dim=num_labels,
    )

    return model


if __name__ == "__main__":
    print(os.getcwd())
    # Create graph from JSON
    paths_file = 'data_preprocessing/preprocessed_datasets/debug_datasets/graph_all_paths.json'
    with open(paths_file, 'r') as f:
        paths_dict_data = json.load(f)
   
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    '''
    Can be generalized to other model & tokenizer later
    '''
    # Define Tokenizer and Model
    batch_size = 8
    eval_batch_size = 2
    num_labels = graph.number_of_nodes() 
    print("num_labels: ", num_labels)
    use_hierarchical_classifier = True
    model_name = 'bert-base-uncased'
    # input_dim = 768
    embedding_dim = num_labels
    uid_to_dimension = set_uid_to_dimension(graph)

    # Check if a GPU is available and use it, otherwise, use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_hierarchical_classifier:
        model = BertWithHierarchicalClassifier(model_name, prediction_target_uids, graph, embedding_dim)
    else:
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        model = BertForSequenceClassification.from_pretrained(model_name, config=config)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f"use_hierarchical_classifier:{use_hierarchical_classifier} --> \nmodel:{model}")

    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head: to fine-tune only the classifier head
    print(model.classifier)
    for param in model.classifier.parameters():
        print(param)
        param.requires_grad = True

    model.to(device)

    # Define Dataset
    dataset_name = 'MVD_100'
    df_path = f'data_preprocessing/preprocessed_datasets/debug_datasets/{dataset_name}.csv'
    # df_path = f'datasets/{dataset_name}.csv'
    max_length = 512
    lr= 1e-3
    num_epoch = 3

    train_df, val_df, test_df = split_dataframe(df_path)
    
    train_encodings = tokenizer(list(train_df["code"]), truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
    val_encodings = tokenizer(list(val_df["code"]), truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
    test_encodings = tokenizer(list(test_df["code"]), truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)

    train_labels = list(train_df["cwe_id"])
    val_labels = list(val_df["cwe_id"])
    test_labels = list(test_df["cwe_id"])
    
    print("uid_to_dimension\n",uid_to_dimension)

    train_dataset = CodeDataset(train_encodings, train_labels, uid_to_dimension)
    val_dataset = CodeDataset(val_encodings, val_labels, uid_to_dimension)
    test_dataset = CodeDataset(test_encodings, test_labels, uid_to_dimension)

    print(len(train_labels),len(val_labels), len(test_labels) )
    def compute_metrics(p):
        print("%%%%%%%%%%%%%%%%INSIDE COMPUTE METRICS")
        predictions, labels = p.predictions, p.label_ids
        # print(f"prediction:{predictions.shape} {type(predictions)}\nlabels:{labels.shape}{type(labels)}")
        # print(f"prediction:{predictions}\nlabels:{labels}")
        pred_dist = model.deembed_dist(predictions) # get probabilities of each nodes
        # print(f"pred_dist: \n{pred_dist}")
        pred_labels = model.dist_to_labels(pred_dist)
        predictions = pred_labels
        print(f"pred_labels:{pred_labels}")
        labels = np.argmax(labels, axis=-1)
        print(f"labels: {labels}")
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Define loss function, optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    callbacks = [EarlyStoppingCallback(patience=2, threshold=0.8)]
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epoch,
        weight_decay=0.01,
        logging_dir='./logs',
        output_dir='./outputs',
        evaluation_strategy="steps",
        eval_steps=5,  
        logging_steps=1,
        learning_rate=lr,
        remove_unused_columns=False,  # Important for our custom loss function
        disable_tqdm=False,
        load_best_model_at_end = True,
        metric_for_best_model = "accuracy",
        greater_is_better = True,
    )

    trainer = CustomTrainer(
        use_hierarchical_classifier = use_hierarchical_classifier,
        model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        model_init=model_init,
        data_collator=data_collator,
        callbacks=callbacks,
       
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=3,
        # compute_objective=compute_objective,
    )

    print(f"best_trial: {best_trial}")
    '''
    trainer.train()
    
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