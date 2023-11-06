import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["WANDB_SILENT"] = "true"

import json
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, TrainingArguments, DataCollatorWithPadding
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertConfig
# from transformers.trainer_callback import EarlyStoppingCallback

import matplotlib.pyplot as plt

from src.trainer import CustomTrainer
from src.dataset import CodeDataset, split_dataframe
from src.graph import create_graph_from_json, set_uid_to_dimension
from src.classifier import get_model_and_tokenizer
from src.callback import EarlyStoppingCallback, WandbCallback, OptunaPruningCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import optuna
from optuna.trial import TrialState
from datasets import load_dataset

import random
import argparse
import wandb

# from sql_db import create_connection
import joblib
from torch.optim import AdamW

torch.cuda.empty_cache()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# Objective function for Optuna
def objective(trial, args):
  
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 1, 32, log=True)
    classifier_factor = trial.suggest_float("classifier_factor",1, 100, log=True)
    # loss_weight = trial.suggest_categorical('loss_weight_method', ['default', 'eqaulize', 'descendants','reachable_leaf_nodes'])
    
    # args.loss_weight = loss_weight # should remove

    # Create graph from JSON
    with open(args.node_paths_dir, 'r') as f:
        paths_dict_data = json.load(f)
   
    # actual targets to be predicted
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    # Define Tokenizer and Model
    num_labels = graph.number_of_nodes() 
    print("num_labels: ", num_labels)
    uid_to_dimension = set_uid_to_dimension(graph)
   
    # Check if a GPU is available and use it, otherwise, use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    model, tokenizer = get_model_and_tokenizer(args, num_labels, prediction_target_uids, graph)
    wandb.watch(model)

    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head: to fine-tune only the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    

    model.to(device)

    # Function to tokenize on the fly
    def encode(example):
        # tokenized_inputs = tokenizer(example['code'], truncation=True, padding=True, max_length=args.max_length,return_tensors="pt").to(device)
        tokenized_inputs = tokenizer(example['code'], truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
        # tokenized_inputs['labels'] = one_hot_encode(example['cwe_id'])
        tokenized_inputs['labels'] = example['cwe_id']
        return tokenized_inputs

    # Load dataset and make huggingface datasts
  
    data_files = {
    'train': f'{args.train_data_dir}',
    'validation': f'{args.val_data_dir}',
    'test': f'{args.test_data_dir}',
    }
    
    dataset = load_dataset('csv', data_files=data_files)
    # Set the transform function for on-the-fly tokenization
    dataset.set_transform(encode)

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    print("TRAIN/VAL/TEST SET LENGTHS:",len(train_dataset), len(val_dataset), len(test_dataset))

    def compute_metrics(p):
        # print("%%%%%%%%%%%%%%%%INSIDE COMPUTE METRICS")

        predictions, labels = p.predictions, p.label_ids
        pred_dist = model.deembed_dist(predictions) # get probabilities of each nodes
        # print(f"pred_dist: \n{pred_dist}")
        if args.use_hierarchical_classifier:
            pred_labels = model.dist_to_cwe_ids(pred_dist)
            predictions = pred_labels
        # print(f"pred_labels:{pred_labels}")
        # print(f"labels: {labels}")
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0.0, labels=prediction_target_uids)
        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {
            "balanced_accuracy":balanced_acc,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

  
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_steps = int(round(args.eval_samples/per_device_train_batch_size, -1))

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        max_steps=eval_steps*args.max_evals,
        weight_decay=weight_decay,
        logging_dir='./logs',
        output_dir='./outputs',
        evaluation_strategy="steps",
        eval_steps=eval_steps,  
        save_steps=eval_steps,
        logging_steps=eval_steps,
        learning_rate=lr,
        remove_unused_columns=False,  # Important for our custom loss function
        disable_tqdm=True,
        load_best_model_at_end = True,
        metric_for_best_model = args.eval_metric,
        greater_is_better = True,
    )

    adam_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    # Parameters of the base model without the classification head
    if args.use_hierarchical_classifier:
        base_params = list(model.model.parameters())
    else:
        base_params = list(model.bert.parameters())
        # Parameters of the classification head
    classifier_params = list(model.classifier.parameters())
    base_lr = lr
    classifier_lr = base_lr*classifier_factor

    optimizer = AdamW([ { "params":  base_params, "lr": base_lr}, {"params": classifier_params, "lr": classifier_lr} ], **adam_kwargs)


    trainer = CustomTrainer(
        use_hierarchical_classifier = args.use_hierarchical_classifier,
        uid_to_dimension = uid_to_dimension,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        callbacks=[OptunaPruningCallback(trial=trial, args=args),WandbCallback],  
        # callbacks=[EarlyStoppingCallback(patience=5, threshold=0),WandbCallback],
    )

    # Train and evaluate the model
    trainer.train()
    metrics = trainer.evaluate()
    # Log metrics to wandb
    wandb.log(metrics)
    print("metrics:",metrics)

    return metrics[f"eval_{args.eval_metric}"]
    

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Hyperparameter optimization using Optuna")

    # Add arguments
    # parser.add_argument('--data-dir', type=str, default='datasets_', help='Path to the dataset directory')
    parser.add_argument('--node-paths-dir', type=str, default='data_preprocessing/preprocessed_datasets/debug_datasets/graph_final_cwe_paths.json', help='Path to the dataset directory')
    parser.add_argument('--train-data-dir', type=str, default='datasets_/train_dataset.csv', help='Path to the train dataset directory')
    parser.add_argument('--val-data-dir', type=str, default='datasets_/balanced_validation_dataset.csv', help='Path to the val dataset directory')
    parser.add_argument('--test-data-dir', type=str, default='datasets_/test_dataset.csv', help='Path to the test dataset directory')
    parser.add_argument('--debug-mode', action='store_true', help='Flag for using small dataset for debug')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased', help='Name of the model to use')
    parser.add_argument('--num-trials', type=int, default=10, help='Number of trials for Optuna')
    parser.add_argument('--use-weight-sampling', action='store_true', help='Flag for using weight sampling')
    parser.add_argument('--use-hierarchical-classifier', action='store_true', help='Flag for hierarchical classification') #--use-hierarchical-classifier --> true
    parser.add_argument('--loss-weight', type=str, default='equalize', help="Loss weight type for Hierarchical classification loss, options: 'default', 'equalize', 'descendants','reachable_leaf_nodes'")
    parser.add_argument('--num-train-epochs', type=int, default=5, help='Number of epoch for training')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum length for token number')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--n-gpu', type=int, default=1, help='Number of GPU')
    parser.add_argument('--study-name', type=str, default='HC_BERT', help='Optuna study name')
    parser.add_argument('--max-evals', type=int, default=500, help='Maximum number of evaluation steps')
    parser.add_argument('--eval-samples', type=int, default=4800, help='Number of training samples between two evaluations. It should be divisible by 32')
    parser.add_argument('--output-dir', type=str, default='outputs', help='HPO output directory')
    parser.add_argument('--eval-metric', type=str, default='f1', help='Evaluation metric')

    # Parse the command line arguments
    args = parser.parse_args()
    if args.debug_mode:
        args.study_name = f"{args.study_name}_s"
        args.train_data_dir = 'datasets_/train_small_data.csv'
        args.test_data_dir = 'datasets_/test_small_data.csv'
    if args.use_hierarchical_classifier:
        args.study_name = f"{args.study_name}_{args.loss_weight}"
    else:
        args.study_name = f"{args.study_name}_CE"
    if args.eval_samples%32:
        raise ValueError(f"--eval-samples {args.eval_samples} is not divisible by 32")

    args.study_name = f"{args.study_name}_max_evals{args.max_evals}"
    print("MAIN - args",args)

    set_seed(args)

    # Initialize a new run
    wandb.init(project="TransVulDet", name=args.study_name)

    print(os.getcwd())

    tpeopts = optuna.samplers.TPESampler.hyperopt_parameters()
    tpeopts.update({'n_startup_trials': 8})

    # Initialize Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=args.max_evals, reduction_factor=3
        ),
        sampler = optuna.samplers.TPESampler(
            **tpeopts
        ),
        storage=f"sqlite:///database/{args.study_name}.db",
        load_if_exists=True,
        
        )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.num_trials, timeout=258500)
    
    # Print results
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best {args.eval_metric}: {study.best_value}")

    location = args.output_dir + "/study.pkl"

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    joblib.dump(study, location)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
