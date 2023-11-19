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

import random
import argparse
import wandb

# from sql_db import create_connection
import joblib
from torch.optim import AdamW
import torch.nn.functional as F
import copy

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
        # Find the index of the max softmax probability
        softmax_idx = np.argmax(pred)
        cwe_id = list(target_to_dimension.keys())[softmax_idx]
        if cwe_id not in list(target_to_dimension.keys()):
            print(f"cwe_id:{cwe_id} is NOT in target_to_dimension!!!!!")
        cwe_target_idx = target_to_dimension[cwe_id]
        pred_labels.append(cwe_target_idx)
    return pred_labels

def mapping_cwe_to_target_label(cwe_label, target_to_dimension):
    # Convert each tensor element to its corresponding dictionary value
    mapped_labels = [target_to_dimension[int(cwe_id)] for cwe_id in cwe_label]
    return mapped_labels

def get_class_weight(df,target_to_dimension):
    cwe_list = df['assignedclass'].tolist()
    idx_classes = [target_to_dimension[int(cwe_id)] for cwe_id in cwe_list]
    class_counts = np.bincount(idx_classes, minlength=len(target_to_dimension))  # Ensure 'minlength' covers all your classes
    # Calculate class weights (inverse of the frequency)
    weights = 1. / class_counts
    weights = weights / weights.sum()  # Normalize to make the sum of weights equal to 1
    weights[class_counts == 0] = 0  # Set weight to 0 if class count is 0
    class_weights = torch.FloatTensor(weights)
    return class_weights

torch.cuda.empty_cache()
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Hyperparameter optimization using Optuna")
# Add arguments
# parser.add_argument('--data-dir', type=str, default='datasets_', help='Path to the dataset directory')
parser.add_argument('--node-paths-dir', type=str, default='data_preprocessing/preprocessed_datasets/debug_datasets/graph_assignedcwe_paths.json', help='Path to the dataset directory')
parser.add_argument('--train-data-dir', type=str, default='datasets_/train_dataset.csv', help='Path to the train dataset directory')
parser.add_argument('--val-data-dir', type=str, default='datasets_/balanced_validation_dataset.csv', help='Path to the val dataset directory')
parser.add_argument('--test-data-dir', type=str, default='datasets_/test_dataset.csv', help='Path to the test dataset directory')
parser.add_argument('--debug-mode', action='store_true', help='Flag for using small dataset for debug')
parser.add_argument('--model-name', type=str, default='bert-base-uncased', help='Name of the model to use')
parser.add_argument('--num-trials', type=int, default=10, help='Number of trials for Optuna')
parser.add_argument('--use-weight-sampling', action='store_true', help='Flag for using weight sampling')
parser.add_argument('--use-hierarchical-classifier', action='store_true', help='Flag for hierarchical classification') #--use-hierarchical-classifier --> true
parser.add_argument('--use-tuning-last-layer', action='store_true', help='Flag for only fine-tuning pooler layer')
parser.add_argument('--loss-weight', type=str, default='equalize', help="Loss weight type for Hierarchical classification loss, options: 'default', 'equalize', 'descendants','reachable_leaf_nodes'")
parser.add_argument('--use-focal-loss', action='store_true', help='Flag for using focal loss instead of cross entropy loss')
parser.add_argument('--num-train-epochs', type=int, default=5, help='Number of epoch for training')
parser.add_argument('--max-length', type=int, default=512, help='Maximum length for token number')
parser.add_argument('--seed', type=int, default=42, help='Seed')
parser.add_argument('--n-gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--study-name', type=str, default='HC_BERT', help='Optuna study name')
parser.add_argument('--max-evals', type=int, default=150, help='Maximum number of evaluation steps')
parser.add_argument('--eval-samples', type=int, default=9600, help='Number of training samples between two evaluations. It should be divisible by 32')
parser.add_argument('--output-dir', type=str, default='outputs', help='HPO output directory')
parser.add_argument('--eval-metric', type=str, default='f1', help='Evaluation metric')
args = parser.parse_args(["--model-name", "microsoft/graphcodebert-base","--num-trials","1","--n-gpu","1","--max-evals","150","--study-name","testtt","--eval-metric","f1","--loss-weight","default","--use-hierarchical-classifier","--use-tuning-last-layer"])
# args = parser.parse_args(["--model-name", "microsoft/codebert-base","--num-trials","1","--n-gpu","1","--max-evals","150","--study-name","testtt","--eval-metric","f1","--loss-weight","default","--use-hierarchical-classifier"])
# args = parser.parse_args(["--model-name", "microsoft/graphcodebert-base","--num-trials","1","--n-gpu","1","--max-evals","150","--study-name","testtt2","--eval-metric","f1","--loss-weight","default"])
# args = parser.parse_args(["--model-name", "microsoft/graphcodebert-base","--num-trials","1","--n-gpu","1","--max-evals","150","--study-name","testtt2","--eval-metric","f1","--loss-weight","default", "--use-tuning-last-layer"])

print("MAIN - args",args)

set_seed(args)
# Suggest hyperparameters
lr = 1e-4
weight_decay = 1e-7
per_device_train_batch_size = 4
classifier_factor = 10

# Create graph from JSON
with open(args.node_paths_dir, 'r') as f:
    paths_dict_data = json.load(f)
# actual targets to be predicted
prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
graph = create_graph_from_json(paths_dict_data, max_depth=None)
# Define Tokenizer and Model
num_labels = graph.number_of_nodes() 
target_to_dimension = {target:idx for idx,target in enumerate(prediction_target_uids)}
print(f"num_all_nodes:{num_labels} num_target_labels: {len(target_to_dimension)}")

# define class weights for focal loss
df = pd.read_csv('datasets_/combined_dataset.csv')
class_weights = get_class_weight(df,target_to_dimension)

# Check if a GPU is available and use it, otherwise, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, tokenizer = get_model_and_tokenizer(args, num_labels, prediction_target_uids, graph)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

model.to(device)

# Function to tokenize on the fly
def encode(example):
    # tokenized_inputs = tokenizer(example['code'], truncation=True, padding=True, max_length=args.max_length,return_tensors="pt").to(device)
    tokenized_inputs = tokenizer(example['code'], truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    # tokenized_inputs['labels'] = one_hot_encode(example['assignedclass'])
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
    print("predictions", len(predictions),predictions[0], predictions[0].shape)
    labels = mapping_cwe_to_target_label(labels, target_to_dimension)
    if args.use_hierarchical_classifier:
        pred_dist = model.deembed_dist(predictions) # get probabilities of each nodes
        pred_cwe_labels = model.dist_to_cwe_ids(pred_dist)
        pred_labels = mapping_cwe_to_target_label(pred_cwe_labels, target_to_dimension)
    else:
        print("predictions", len(predictions), predictions)
        pred_labels = map_predictions_to_target_labels(predictions, target_to_dimension)
      
    predictions = pred_labels
    print(f"predictions:{len(predictions)}{predictions}")
    print(f"labels: {len(labels)}{labels}")
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0.0, labels=list(target_to_dimension.values()))
    acc = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    return {
        "balanced_accuracy":balanced_acc,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# eval_steps = int(round(args.eval_samples/per_device_train_batch_size, -1))
args.max_evals = 1
eval_steps = int(round(args.eval_samples/per_device_train_batch_size, -1)) 
training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    max_steps=eval_steps*args.max_evals,
    weight_decay=weight_decay,
    logging_dir='./logs',
    output_dir='./outputs',
    evaluation_strategy="steps",
    eval_steps=2400,  
    save_steps=2400,
    logging_steps=24,
    learning_rate=lr,
    remove_unused_columns=False,  # Important for our custom loss function
    disable_tqdm=False,
    load_best_model_at_end = True,
    metric_for_best_model = args.eval_metric,
    greater_is_better = True,
)

adam_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
# Parameters of the base model without the classification head
# if args.use_hierarchical_classifier:
#     base_params = list(model.model.parameters())
# else:
#     base_params = [p for n, p in model.named_parameters() if 'classifier' not in n]

if args.use_tuning_last_layer:
    pooler_params_names = [n for n, p in model.named_parameters() if 'pooler' in n]
    base_params_names = [n for n, p in model.named_parameters() if 'encoder.layer.11.output' in n]
    print("pooler_params Name:\n",pooler_params_names)
    print("base_params Name:\n",base_params_names)
    base_params = [p for n, p in model.named_parameters() if 'encoder.layer.11.output' in n]
    if args.use_hierarchical_classifier:
        pooler_params = [p for n, p in model.named_parameters() if 'pooler' in n]
        base_params.append(pooler_params)
else:
    base_params_names = [n for n, p in model.named_parameters() if 'classifier' not in n]
    print("base_params_names:\n",base_params_names)
    base_params = [p for n, p in model.named_parameters() if 'classifier' not in n]

temp_params = []
for param in base_params:
    if type(param) is list:
        for p in param:
            p.requires_grad = True
            temp_params.append(p)
    else:
        param.requires_grad = True
        temp_params.append(param)
print("temp_params",temp_params)
base_params = temp_params

classifier_params_names = [n for n, p in model.classifier.named_parameters()] #['dense.weight', 'dense.bias', 'out_proj.weight', 'out_proj.bias']
print("classifier_params_names",classifier_params_names)
classifier_params = list(model.classifier.parameters())
print("classifier_params", classifier_params)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


base_lr = lr/classifier_factor
classifier_lr = lr
optimizer = AdamW([ { "params":  base_params, "lr": base_lr}, {"params": classifier_params, "lr": classifier_lr} ], **adam_kwargs)
trainer = CustomTrainer(
    use_hierarchical_classifier = args.use_hierarchical_classifier,
    use_focal_loss = args.use_focal_loss,
    prediction_target_uids = prediction_target_uids,
    class_weights = class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),
)


print("BEFORE Train")
# Assuming 'model' is your PyTorch model
# Combine into a dictionary
param_dict = {name: param for name, param in zip(base_params_names, base_params)}
for name, param in zip(classifier_params_names, classifier_params):
    param_dict[name] = param 
initial_state_dict = copy.deepcopy(param_dict)

trainer.train()

param_dict = {name: param for name, param in zip(base_params_names, base_params)}
for name, param in zip(classifier_params_names, classifier_params):
    param_dict[name] = param 
after_state_dict = copy.deepcopy(param_dict)

initial_state_dict = [initial_state_dict.values()]
after_state_list = [after_state_dict.values()]

for i in range(len(initial_state_dict)):
    if initial_state_dict[i] == after_state_list[i]:
        print(f"initial_state_dict[{i}] == after_state_list[{i}]")


# base_params[0]
# base_params[190]
# base_params[197]
# classifier_params[0]
# classifier_params[-1]

# data1 = val_dataset[10:11]
# data2 = val_dataset[11:12]
# data1
# data2
# output1 = model.model(data1.input_ids.to(device), data1.attention_mask.to(device))
# output2 = model.model(data2.input_ids.to(device), data2.attention_mask.to(device))

# logits1 = output1.last_hidden_state
# logits2 = output2.last_hidden_state

# logits1
# logits2

# softmax_logits1 = F.softmax(logits1, dim=-1)
# softmax_logits2 = F.softmax(logits2, dim=-1)

# small_set = val_dataset[:5]
# small_set2 = val_dataset[5:10]

# output1 = model.model(small_set.input_ids.to(device), small_set.attention_mask.to(device)) #BaseModelOutputWithPoolingAndCrossAttentions
# output2 = model.model(small_set2.input_ids.to(device), small_set2.attention_mask.to(device))

# cls_output1 = output1.last_hidden_state[:,0,:] # [5, 768]
# cls_output2 = output2.last_hidden_state[:,0,:]

# cls_output1
# cls_output2

# logits1= model.classifier(cls_output1) # [5, 32]
# logits1

# soft_logits1 =  F.softmax(logits1, dim=-1) # [5, 32]

# Train and evaluate the model
trainer.train()
metrics = trainer.evaluate()
# Log metrics to wandb
wandb.log(metrics)
print("metrics:",metrics)

metrics[f"eval_{args.eval_metric}"]


# Parse the command line arguments
# if args.debug_mode:
#     args.study_name = f"{args.study_name}_debug"
#     args.train_data_dir = 'datasets_/2nd_latest_datasets/train_small_data.csv'
#     args.test_data_dir = 'datasets_/2nd_latest_datasets/test_small_data.csv'
#     args.val_data_dir = 'datasets_/2nd_latest_datasets/val_small_data.csv'
# if args.use_hierarchical_classifier:
#     args.study_name = f"{args.study_name}_{args.loss_weight}"
# else:
#     if args.use_focal_loss:
#         args.study_name = f"{args.study_name}_FL"
#     else:
#         args.study_name = f"{args.study_name}_CE"
# if args.eval_samples%32:
#     raise ValueError(f"--eval-samples {args.eval_samples} is not divisible by 32")

# args.study_name = f"{args.study_name}_max_evals{args.max_evals}"


# # Initialize a new run
# wandb.init(project="TransVulDet", name=args.study_name)

# print(os.getcwd())

# tpeopts = optuna.samplers.TPESampler.hyperopt_parameters()
# tpeopts.update({'n_startup_trials': 8})

# # Initialize Optuna study
# study = optuna.create_study(
#     study_name=args.study_name,
#     direction="maximize",
#     pruner=optuna.pruners.HyperbandPruner(
#         min_resource=1, max_resource=args.max_evals, reduction_factor=3
#     ),
#     sampler = optuna.samplers.TPESampler(
#         **tpeopts
#     ),
#     storage=f"sqlite:///database/{args.study_name}.db",
#     load_if_exists=True,
    
#     )
# study.optimize(lambda trial: objective(trial, args), n_trials=args.num_trials, timeout=258500)

# # Print results
# print(f"Best trial: {study.best_trial.params}")
# print(f"Best {args.eval_metric}: {study.best_value}")

# location = args.output_dir + "/study.pkl"

# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# joblib.dump(study, location)

# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))
