import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["WANDB_SILENT"] = "true"

import json
from transformers import TrainingArguments
import matplotlib.pyplot as plt

from src.trainer import CustomTrainer
# from src.dataset import CodeDataset, split_dataframe
from src.graph import create_graph_from_json
from src.classifier import get_model_and_tokenizer
from src.callback import EarlyStoppingCallback, WandbCallback, OptunaPruningCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import optuna
from optuna.trial import TrialState
from datasets import load_dataset

import secrets
import base64
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

def map_predictions_to_target_labels(predictions, target_to_dimension):
    pred_labels = []
    for pred in predictions:
        softmax_idx = np.argmax(pred)
        cwe_id = list(target_to_dimension.keys())[softmax_idx]
        if cwe_id not in list(target_to_dimension.keys()):
            print(f"cwe_id:{cwe_id} is NOT in target_to_dimension!!!!!")
        cwe_target_idx = target_to_dimension[cwe_id]
        pred_labels.append(cwe_target_idx)

    return pred_labels

def mapping_cwe_to_target_label(cwe_label, target_to_dimension):
        mapped_labels = [target_to_dimension[int(cwe_id)] for cwe_id in cwe_label]
        return mapped_labels

def get_class_weight(df,target_to_dimension):
    cwe_list = df['assignedclass'].tolist()
    idx_classes = [target_to_dimension[int(cwe_id)] for cwe_id in cwe_list]
    class_counts = np.bincount(idx_classes, minlength=len(target_to_dimension))  # Ensure 'minlength' covers all classes
    # Calculate class weights (inverse class frequency)
    weights = 1. / class_counts
    weights = weights / weights.sum()  # Normalize to make the sum of weights equal to 1
    weights[class_counts == 0] = 0  # Set weight to 0 if class count is 0
    class_weights = torch.FloatTensor(weights)
    return class_weights

def objective(trial, args):
    lr = trial.suggest_float("classifier_learning_rate", 1e-5, 1e-1, log=True)
    classifier_factor = trial.suggest_float("classifier_factor", 1e1, 1e5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 32, log=True)
    per_device_train_batch_size = 32

    # Create graph from JSON
    with open(args.node_paths_dir, 'r') as f:
        paths_dict_data = json.load(f)
   
    # actual targets to be predicted
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
    target_to_dimension = {target:idx for idx,target in enumerate(prediction_target_uids)}
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    # Define Tokenizer and Model
    num_labels = graph.number_of_nodes() 

    # define class weights for focal loss
    df = pd.read_csv('datasets_/combined_dataset.csv')
    class_weights = get_class_weight(df,target_to_dimension)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    model, tokenizer = get_model_and_tokenizer(args, prediction_target_uids, graph)
    wandb.watch(model)

    # unfreeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = True

    # Unfreeze the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    # Function to tokenize on the fly
    def encode(example):
        tokenized_inputs = tokenizer(example['code'], truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
        tokenized_inputs['labels'] = example['assignedclass']
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

        predictions, labels = p.predictions, p.label_ids
        labels = mapping_cwe_to_target_label(labels, target_to_dimension)
        
        if args.use_hierarchical_classifier:
            pred_dist = model.deembed_dist(predictions) # get probabilities of each nodes
            pred_cwe_labels = model.dist_to_cwe_ids(pred_dist)
            pred_labels = mapping_cwe_to_target_label(pred_cwe_labels, target_to_dimension)
            
        else:
            pred_labels = map_predictions_to_target_labels(predictions, target_to_dimension)
        
        predictions = pred_labels

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=args.eval_metric_average, zero_division=0.0, labels=list(target_to_dimension.values()))
        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {
            "balanced_accuracy":balanced_acc,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    eval_steps = int(round(args.eval_samples/(per_device_train_batch_size*gradient_accumulation_steps), -1))

    # Create the new directory to avoid to save the model checkpoints to the existing folders
    random_str = base64.b64encode(secrets.token_bytes(12)).decode()
    args.output_dir = f'{args.output_dir}/{args.study_name}_{random_str}'
    args.logging_dir = f'{args.logging_dir}/{args.study_name}_{random_str}'
    os.makedirs(args.output_dir) # exist_ok = False
    os.makedirs(args.logging_dir) # exist_ok = False
    
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=32,
        max_steps=eval_steps*args.max_evals,
        weight_decay=weight_decay,
        logging_dir=args.logging_dir,
        output_dir=args.output_dir,
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
    
    # define different learning rate for pre-trained model and classifier
    base_lr = lr/classifier_factor
    if args.debug_mode:
        optimizer = AdamW(model.parameters(), lr=2e-5)

    else:
        classifier_params = list(model.classifier.parameters())
        base_params_names = [n for n, p in model.named_parameters() if 'classifier' not in n]
        base_params = [p for n, p in model.named_parameters() if 'classifier' not in n] 
        if args.use_bilstm:
            bilstm_params = list(model.bilstm.parameters())
            bilistm_lr = trial.suggest_float("BiLSTM_learning_rate", 1e-5, 1e-1, log=True)
            base_params = list(model.model.parameters())
            optimizer = AdamW([ { "params":  base_params, "lr": base_lr}, {"params": bilstm_params, "lr": bilistm_lr}, {"params": classifier_params, "lr": lr} ], **adam_kwargs)

        else:
            optimizer = AdamW([ { "params":  base_params, "lr": base_lr}, {"params": classifier_params, "lr": lr} ], **adam_kwargs)

    trainer = CustomTrainer(
        use_hierarchical_classifier = args.use_hierarchical_classifier,
        prediction_target_uids = prediction_target_uids,
        use_focal_loss = args.use_focal_loss,
        use_bilstm = args.use_bilstm,
        class_weights = class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        callbacks=[OptunaPruningCallback(trial=trial, args=args),WandbCallback],  
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

    parser = argparse.ArgumentParser(description="Hyperparameter optimization using Optuna")

    parser.add_argument('--node-paths-dir', type=str, default='data_preprocessing/preprocessed_datasets/debug_datasets/graph_assignedcwe_paths.json', help='Path to the dataset directory')
    parser.add_argument('--train-data-dir', type=str, default='datasets_/train_dataset.csv', help='Path to the train dataset directory')
    parser.add_argument('--val-data-dir', type=str, default='datasets_/balanced_validation_dataset.csv', help='Path to the val dataset directory')
    parser.add_argument('--test-data-dir', type=str, default='datasets_/test_dataset.csv', help='Path to the test dataset directory')
    parser.add_argument('--debug-mode', action='store_true', help='Flag for using small dataset for debug')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased', help='Name of the model to use')
    parser.add_argument('--num-trials', type=int, default=1, help='Number of trials for Optuna')
    parser.add_argument('--use-bilstm', action='store_true', help='Flag for BiLSTM with Transformer Model')
    parser.add_argument('--use-hierarchical-classifier', action='store_true', help='Flag for hierarchical classification') #--use-hierarchical-classifier --> true
    parser.add_argument('--use-tuning-last-layer', action='store_true', help='Flag for only fine-tuning pooler layer among base model layers')
    parser.add_argument('--use-tuning-classifier', action='store_true', help='Flag for only fine-tuning classifier')
    parser.add_argument('--loss-weight', type=str, default='equalize', help="Loss weight type for Hierarchical classification loss, options: 'default', 'equalize', 'descendants','reachable_leaf_nodes'")
    parser.add_argument('--use-focal-loss', action='store_true', help='Flag for using focal loss instead of cross entropy loss')
    parser.add_argument('--direction', type=str, default='maximize', help='Direction to optimize')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum length for token number')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--n-gpu', type=int, default=1, help='Number of GPU')
    parser.add_argument('--study-name', type=str, default='2311_FT', help='Optuna study name')
    parser.add_argument('--max-evals', type=int, default=9, help='Maximum number of evaluation steps')
    parser.add_argument('--eval-samples', type=int, default=40960, help='Number of training samples between two evaluations. It should be divisible by 32')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='HPO output directory')
    parser.add_argument('--logging-dir', type=str, default='./logs', help='Trainer log directory')
    parser.add_argument('--eval-metric', type=str, default='f1', help='Evaluation metric')
    parser.add_argument('--eval-metric-average', type=str, default='macro', help='Evaluation metric average')

    args = parser.parse_args()

    args.study_name = f"{args.study_name}_{args.eval_metric}"
    # if args.debug_mode:
    #     args.study_name = f"{args.study_name}_debug"
    #     args.train_data_dir = 'datasets_/2nd_latest_datasets/train_small_data.csv'
    #     args.test_data_dir = 'datasets_/2nd_latest_datasets/test_small_data.csv'
    #     args.val_data_dir = 'datasets_/2nd_latest_datasets/val_small_data.csv'

    if not args.use_tuning_classifier:
        if args.use_tuning_last_layer:
            args.study_name = f"{args.study_name}_ll"
        else:
            args.study_name = f"{args.study_name}"
    else:
        args.study_name = f"{args.study_name}_cls"

    if args.use_hierarchical_classifier:
        args.study_name = f"{args.study_name}_{args.loss_weight}"
    else:
        if args.use_focal_loss:
            args.study_name = f"{args.study_name}_FL"
        else:
            args.study_name = f"{args.study_name}_CE"


    if args.eval_samples%32:
        raise ValueError(f"--eval-samples {args.eval_samples} is not divisible by 32")

    args.study_name = f"{args.study_name}_max_evals{args.max_evals}_samples{args.eval_samples}"

    print("MAIN - args",args)

    set_seed(args)

    # Initialize a new run
    wandb.init(project="TransVulDet", name=args.study_name)

    tpeopts = optuna.samplers.TPESampler.hyperopt_parameters()
    tpeopts.update({'n_startup_trials': 8})

    # Initialize Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
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

    location = f"{args.output_dir}/study.pkl"

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    joblib.dump(study, location)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))