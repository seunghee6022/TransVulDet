import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["WANDB_SILENT"] = "true"

import json
from transformers import TrainingArguments

from src.trainer_debug import CustomTrainer
# from src.dataset import CodeDataset, split_dataframe
from src.graph import create_graph_from_json
from src.classifier_debug import get_model_and_tokenizer
from src.callback import EarlyStoppingCallback, WandbCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score

from datasets import load_dataset

import secrets
import base64
import random
import argparse
import wandb
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

def train(args, best_param):
    lr = best_param['classifier_learning_rate']
    classifier_factor = best_param['classifier_factor']
    weight_decay = best_param['weight_decay']
    gradient_accumulation_steps = best_param['gradient_accumulation_steps']
    if args.use_bilstm:
        bilistm_lr = best_param['BiLSTM_learning_rate'] if best_param['BiLSTM_learning_rate'] else 2e-5
    per_device_train_batch_size = 32

    # Create graph from JSON
    with open(args.node_paths_dir, 'r') as f:
        paths_dict_data = json.load(f)
   

    ith open(args.node_paths_dir, 'r') as f:
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
    os.makedirs(args.output_dir) # exist_ok = False
    os.makedirs(args.logging_dir)

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=32,
        max_steps=eval_steps*args.max_evals,
        do_train=True,  # Set to True to perform training
        do_eval=True,  
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

    base_lr = lr/classifier_factor
    
    if args.debug_mode:
        optimizer = AdamW(model.parameters(), lr=2e-5)

    else:
        classifier_params = list(model.classifier.parameters())
        base_params_names = [n for n, p in model.named_parameters() if 'classifier' not in n]
        base_params = [p for n, p in model.named_parameters() if 'classifier' not in n] 
        if args.use_bilstm:
            bilstm_params = list(model.bilstm.parameters())
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
        callbacks=[WandbCallback],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    # Log metrics to wandb
    wandb.log(eval_metrics)
    
    return eval_metrics

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Hyperparameter optimization using Optuna")

    parser.add_argument('--node-paths-dir', type=str, default='data_preprocessing/preprocessed_datasets/debug_datasets/graph_assignedcwe_paths.json', help='Path to the dataset directory')
    parser.add_argument('--train-data-dir', type=str, default='datasets_/train_dataset.csv', help='Path to the train dataset directory')
    parser.add_argument('--val-data-dir', type=str, default='datasets_/validation_dataset.csv', help='Path to the val dataset directory')
    parser.add_argument('--test-data-dir', type=str, default='datasets_/test_dataset.csv', help='Path to the test dataset directory')
    parser.add_argument('--debug-mode', action='store_true', help='Flag for using small dataset for debug')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased', help='Name of the model to use')
    parser.add_argument('--num-trials', type=int, default=10, help='Number of trials for Optuna')
    parser.add_argument('--use-bilstm', action='store_true', help='Flag for BiLSTM with Transformer Model')
    parser.add_argument('--use-weight-sampling', action='store_true', help='Flag for using weight sampling')
    parser.add_argument('--use-hierarchical-classifier', action='store_true', help='Flag for hierarchical classification') 
    parser.add_argument('--use-tuning-last-layer', action='store_true', help='Flag for only fine-tuning pooler layer among base model layers')
    parser.add_argument('--use-tuning-classifier', action='store_true', help='Flag for only fine-tuning classifier')
    parser.add_argument('--loss-weight', type=str, default='equalize', help="Loss weight type for Hierarchical classification loss, options: 'default', 'equalize', 'descendants','reachable_leaf_nodes'")
    parser.add_argument('--use-focal-loss', action='store_true', help='Flag for using focal loss instead of cross entropy loss')
    parser.add_argument('--direction', type=str, default='maximize', help='Direction to optimize')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum length for token number')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--n-gpu', type=int, default=1, help='Number of GPU')
    parser.add_argument('--study-name', type=str, default='Train', help='Optuna study name')
    parser.add_argument('--max-evals', type=int, default=5000, help='Maximum number of evaluation steps')
    parser.add_argument('--eval-samples', type=int, default=40960, help='Number of training samples between two evaluations. It should be divisible by 32')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Trainer output directory')
    parser.add_argument('--logging-dir', type=str, default='./logs', help='Trainer logging directory')
    parser.add_argument('--eval-metric', type=str, default='f1', help='Evaluation metric')
    parser.add_argument('--eval-metric-average', type=str, default='weighted', help='Evaluation metric average')

    # Parse the command line arguments
    args = parser.parse_args()

    # this best params can get from run visualization/HPO_visualization.ipynb
    best_param_list =  {
            'CodeBERT+BiLSTM_f1_CE' : {'BiLSTM_learning_rate': 0.034008690972232435, 'classifier_factor': 139.68095332275686, 'classifier_learning_rate': 0.01160917664931142, 'gradient_accumulation_steps': 2, 'weight_decay': 9.020412876278314e-06},
            'CodeBERT+BiLSTM_f1_FL' : {'BiLSTM_learning_rate': 0.000318572916799483, 'classifier_factor': 24.575571392567063, 'classifier_learning_rate': 0.00029183324213769403, 'gradient_accumulation_steps': 6, 'weight_decay': 1.59631343579501e-05},
            'CodeBERT+BiLSTM_f1_default' : {'BiLSTM_learning_rate': 1.279877085591794e-05, 'classifier_factor': 80.72263084661715, 'classifier_learning_rate': 0.013402752552511171, 'gradient_accumulation_steps': 3, 'weight_decay': 7.925932464857802e-05},
            'CodeBERT+BiLSTM_f1_descendants' : {'BiLSTM_learning_rate': 0.0003970326603219823, 'classifier_factor': 338.87758137094085, 'classifier_learning_rate': 0.05069880896411844, 'gradient_accumulation_steps': 7, 'weight_decay': 4.2414186104451807e-05},
            'CodeBERT+BiLSTM_f1_equalize' : {'BiLSTM_learning_rate': 0.0003220877921377328, 'classifier_factor': 850.7065378882156, 'classifier_learning_rate': 0.04929124236218869, 'gradient_accumulation_steps': 3, 'weight_decay': 9.238151344811795e-06},
            'CodeBERT+BiLSTM_f1_reachable_leaf_nodes' : {'BiLSTM_learning_rate': 8.17950823933379e-05, 'classifier_factor': 29.262907091908694, 'classifier_learning_rate': 0.006060646821315373, 'gradient_accumulation_steps': 5, 'weight_decay': 8.955364128624764e-06},
            'GraphCodeBERT+BiLSTM_f1_CE' : {'BiLSTM_learning_rate': 1.3660252213946701e-05, 'classifier_factor': 665.2802919644846, 'classifier_learning_rate': 0.0007258867242144063, 'gradient_accumulation_steps': 2, 'weight_decay': 1.0449154163675762e-05},
            'GraphCodeBERT+BiLSTM_f1_FL' : {'BiLSTM_learning_rate': 0.007354054646214738, 'classifier_factor': 93.78208233336159, 'classifier_learning_rate': 5.53498593117409e-05, 'gradient_accumulation_steps': 8, 'weight_decay': 4.5844220046049055e-06},
            'GraphCodeBERT+BiLSTM_f1_default' : {'BiLSTM_learning_rate': 0.00022919638718472263, 'classifier_factor': 1139.6003923638746, 'classifier_learning_rate': 0.05434920715623826, 'gradient_accumulation_steps': 1, 'weight_decay': 8.458353951755318e-06},
            'GraphCodeBERT+BiLSTM_f1_descendants' : {'BiLSTM_learning_rate': 0.00018875583923956264, 'classifier_factor': 99.50070477528486, 'classifier_learning_rate': 0.016943619592600573, 'gradient_accumulation_steps': 10, 'weight_decay': 0.0001161442879113915},
            'GraphCodeBERT+BiLSTM_f1_equalize' : {'BiLSTM_learning_rate': 1.3549235554413088e-05, 'classifier_factor': 199.31269483865094, 'classifier_learning_rate': 0.04752125242604547, 'gradient_accumulation_steps': 13, 'weight_decay': 1.781595238815297e-06},
            'GraphCodeBERT+BiLSTM_f1_reachable_leaf_nodes' : {'BiLSTM_learning_rate': 7.770294742008388e-05, 'classifier_factor': 179.26622057119513, 'classifier_learning_rate': 0.01998194431093307, 'gradient_accumulation_steps': 3, 'weight_decay': 2.2905786363408954e-07},

        }
    args.study_name = f"{args.study_name}_{args.eval_metric}"
    if args.debug_mode:
        args.study_name = f"{args.study_name}_debug"
        args.train_data_dir = 'datasets_/2nd_latest_datasets/train_small_data.csv'
        args.test_data_dir = 'datasets_/2nd_latest_datasets/test_small_data.csv'
        args.val_data_dir = 'datasets_/2nd_latest_datasets/val_small_data.csv'

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

    pamram_key = args.study_name
    if pamram_key in best_param_list.keys():
        best_param = best_param_list[pamram_key]
        print(pamram_key, best_param)
    else:
        print("This key",pamram_key,"is not in best_param_list")
    
    if args.eval_samples%32:
        raise ValueError(f"--eval-samples {args.eval_samples} is not divisible by 32")

    random_str = base64.b64encode(secrets.token_bytes(12)).decode()
    args.output_dir = f'./outputs/{args.study_name}_{random_str}'
    args.logging_dir = f'./logs/{args.study_name}_{random_str}'

    args.study_name = f"{args.study_name}_max_evals{args.max_evals}_samples{args.eval_samples}_{random_str}"

    set_seed(args)

    wandb.init(project="TransVulDet", name=args.study_name)
    print("MAIN - args",args)
    eval_metrics = train(args,best_param)
    print("MAIN after training - args",args)
  
 