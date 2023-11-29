import os
import torch
import json
from datasets import load_dataset
from transformers import AutoModel
from src.graph import create_graph_from_json
from src.classifier_debug import get_model_and_tokenizer
from src.dataset import vulDataset
# from main_hpo_sqlite_debug import mapping_cwe_to_target_label, map_predictions_to_target_labels
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import subprocess


def map_predictions_to_target_labels(predictions, target_to_dimension):
    # print("predictions",predictions.shape)
    pred_labels = []
    # print("predictions",len(predictions),predictions)
    for pred in predictions:
        # Find the index of the max softmax probability
        softmax_idx = np.argmax(pred)
        # print(softmax_idx)
        cwe_id = list(target_to_dimension.keys())[softmax_idx]
        # print("softmax_idx:",softmax_idx, "pred:",pred, "cwe_id",cwe_id)
        if cwe_id not in list(target_to_dimension.keys()):
            print(f"cwe_id:{cwe_id} is NOT in target_to_dimension!!!!!")
        cwe_target_idx = target_to_dimension[cwe_id]
        pred_labels.append(cwe_target_idx)

    return pred_labels

def mapping_cwe_to_target_label(cwe_label, target_to_dimension):
        # Convert each tensor element to its corresponding dictionary value
        mapped_labels = [target_to_dimension[int(cwe_id)] for cwe_id in cwe_label]
        return mapped_labels

def load_best_model_with_best_accuracy(args):
    # Find the last checkpoint folder
    last_checkpoint_folder = None
    for folder_name in sorted(os.listdir(args.checkpoint_dir), reverse=True):
        if folder_name.startswith("checkpoint-"):
            last_checkpoint_folder = folder_name
            break

    if last_checkpoint_folder is None:
        print("No checkpoint folders found.")
        return None

    # Construct the path to the trainer_state.json in the last checkpoint folder
    state_file_path = os.path.join(args.checkpoint_dir, last_checkpoint_folder, 'trainer_state.json')

    # Read the trainer_state.json file
    try:
        with open(state_file_path, 'r') as file:
            state_data = json.load(file)
    except FileNotFoundError:
        print(f"No trainer_state.json found in {last_checkpoint_folder}.")
        return None

    # Extract the path to the best model checkpoint
    best_model_path = state_data.get('best_model_checkpoint')
    # print("best_model_path",best_model_path)
    if best_model_path:
        print(f"Best model found at: {best_model_path}")

    # best_model_path = os.path.join(checkpoint_directory, 'path_to_best_model_checkpoint')

    config = RobertaConfig()
    model = RobertaModel(config)
   
    # Create graph from JSON
    with open(args.node_paths_dir, 'r') as f:
        paths_dict_data = json.load(f)
   
    # actual targets to be predicted
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
    # print("prediction_target_uids",prediction_target_uids)
    # target_to_dimension = {target:idx for idx,target in enumerate(prediction_target_uids)}
    # print("target_to_dimension", target_to_dimension)
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    # Check if a GPU is available and use it, otherwise, use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    model, tokenizer = get_model_and_tokenizer(args, prediction_target_uids, graph)
    state_dict = torch.load(f'{best_model_path}/pytorch_model.bin', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(model)
    return model, tokenizer

def find_best_checkpoint_by_macro_f1(args):
    # Find the last checkpoint folder
    last_checkpoint_folder = None
    for folder_name in sorted(os.listdir(args.checkpoint_dir), reverse=True):
        if folder_name.startswith("checkpoint-"):
            last_checkpoint_folder = folder_name
            break

    if last_checkpoint_folder is None:
        print("No checkpoint folders found.")
        return None

    # Construct the path to the trainer_state.json in the last checkpoint folder
    state_file_path = os.path.join(args.checkpoint_dir, last_checkpoint_folder, 'trainer_state.json')

    # Read the trainer_state.json file
    try:
        with open(state_file_path, 'r') as file:
            trainer_state = json.load(file)
    except FileNotFoundError:
        print(f"No trainer_state.json found in {last_checkpoint_folder}.")
        return None

    best_macro_f1 = 0
    best_checkpoint = None

    # Iterate through the log history
    for entry in trainer_state["log_history"]:
        # Check if 'eval_macro_f1' is in the entry
        if "eval_macro_f1" in entry:
            macro_f1 = entry["eval_macro_f1"]
            step = entry["step"]
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_checkpoint = f"checkpoint-{step}"
                print(f'best_macro_f1:{best_macro_f1}  | best_balanced:{entry["eval_balanced_accuracy"]} | best_f1:{entry["eval_f1"]}| best_acc:{entry["eval_accuracy"]} | best_checkpoint:{best_checkpoint}')
            if f"checkpoint-{step}" == last_checkpoint_folder:
                print(f'last_macro_f1: {entry["eval_macro_f1"]}  | last_balanced:{entry["eval_balanced_accuracy"]} | last_f1:{entry["eval_f1"]}| last_acc:{entry["eval_accuracy"]} | last_checkpoint:{last_checkpoint_folder}')
                
    # best_model_path = f'{args.checkpoint_dir}/{best_checkpoint}'
    best_model_path = f'{args.checkpoint_dir}/{last_checkpoint_folder}'
    print("MARCO F1 -- best_model_path",best_model_path)
    # print("MARCO F1 -- last_model_path",best_model_path)

    config = RobertaConfig()
    model = RobertaModel(config)
   
    # Create graph from JSON
    with open(args.node_paths_dir, 'r') as f:
        paths_dict_data = json.load(f)
   
    # actual targets to be predicted
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()] # 204
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    # Check if a GPU is available and use it, otherwise, use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    model, tokenizer = get_model_and_tokenizer(args, prediction_target_uids, graph)
    state_dict = torch.load(f'{best_model_path}/pytorch_model.bin', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    # print(model)
    return model, tokenizer

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print("This is load_best_model_and_compute_metric.py")
 
    parser = argparse.ArgumentParser(description="Fine-tuning Models")

    # Add arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./outputs', help='Path to the checkpoint directory')
    parser.add_argument('--node-paths-dir', type=str, default='data_preprocessing/preprocessed_datasets/debug_datasets/graph_assignedcwe_paths.json', help='Path to the dataset directory')
    parser.add_argument('--test-data-dir', type=str, default='datasets_/test_dataset.csv', help='Path to the test dataset directory')
    parser.add_argument('--debug-mode', action='store_true', help='Flag for using small dataset for debug')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased', help='Name of the model to use')
    parser.add_argument('--use-bilstm', action='store_true', help='Flag for BiLSTM with Transformer Model')
    parser.add_argument('--loss-weight', type=str, default='equalize', help="Loss weight type for Hierarchical classification loss, options: 'default', 'equalize', 'descendants','reachable_leaf_nodes'")
    parser.add_argument('--use-focal-loss', action='store_true', help='Flag for using focal loss instead of cross entropy loss')
    parser.add_argument('--use-hierarchical-classifier', action='store_true', help='Flag for hierarchical classification') #--use-hierarchical-classifier --> true
    parser.add_argument('--max-length', type=int, default=512, help='Maximum length for token number')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--n-gpu', type=int, default=1, help='Number of GPU')
    # parser.add_argument('--study-name', type=str, default='2311_FT', help='Optuna study name')
    # parser.add_argument('--max-evals', type=int, default=9, help='Maximum number of evaluation steps')
    # parser.add_argument('--eval-samples', type=int, default=40960, help='Number of training samples between two evaluations. It should be divisible by 32')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='HPO output directory')
    parser.add_argument('--logging-dir', type=str, default='./logs', help='Trainer log directory')
    parser.add_argument('--eval-metric', type=str, default='f1', help='Evaluation metric')
    # parser.add_argument('--eval-metric-average', type=str, default='macro', help='Evaluation metric average')

    # Parse the command line arguments
    args = parser.parse_args()
  
    print("args.use_hierarchical_classifier | args.use_bilstm",args.use_hierarchical_classifier, args.use_bilstm)
    if 'Graph' in args.checkpoint_dir:
        args.model_name = "microsoft/graphcodebert-base" 
    else:
        args.model_name = "microsoft/codebert-base" 
    
    print("args.checkpoint_dir)",args.checkpoint_dir)
    print("Args",args)

    # Create graph from JSON
    with open(args.node_paths_dir, 'r') as f:
        paths_dict_data = json.load(f)
   
    # actual targets to be predicted
    prediction_target_uids = [int(key) for key in paths_dict_data.keys()]
    # print("prediction_target_uids",prediction_target_uids)
    target_to_dimension = {target:idx for idx,target in enumerate(prediction_target_uids)}
    # print("target_to_dimension", target_to_dimension)
    graph = create_graph_from_json(paths_dict_data, max_depth=None)


    # acc_model, acc_tokenizer = load_best_model_with_best_accuracy(args)
    model, tokenizer = find_best_checkpoint_by_macro_f1(args)
   
    
    def compute_metrics(predictions, labels):
        print("%%%%%%%%%%%%%%%%INSIDE COMPUTE METRICS")
        print("Initial argmax idx prediction", predictions[:10], "target_to_dimension", target_to_dimension)
        
        # print("[compute_metrics]p.label_ids before mapping_cwe_to_target_label", p.label_ids)
        labels = mapping_cwe_to_target_label(labels, target_to_dimension)
        print("[compute_metrics] @@@@@@@ labels [:30]", labels[:30])
        
        if args.use_hierarchical_classifier:
            # prediction value is maxarg index from all nodes
            # dim_to_cwe = {v:k for k,v in model.uid_to_dimension.items()}
            # predictions = [dim_to_cwe[pred] for pred in predictions]
            print("prediction", predictions[:10], "target_to_dimension", target_to_dimension)
    
            pred_dist = model.deembed_dist(predictions) # get probabilities of each nodes
            # print("[if args.use_hierarchical_classifier]pred_dist",pred_dist)
            pred_cwe_labels = model.dist_to_cwe_ids(pred_dist)
            # print("[if args.use_hierarchical_classifier]pred_cwe_labels",pred_cwe_labels)
            pred_labels = mapping_cwe_to_target_label(pred_cwe_labels, target_to_dimension)
            print(f"[Hierarchical Classifier]Unique value: {len(set(pred_labels))} @@@@@@@ pred_labels[:30]:{pred_labels[:30]}")
        else:
            pred_labels = predictions
            print("prediction", predictions[:10], "target_to_dimension", prediction_target_uids)
    
            # pred_labels = [prediction_target_uids[pred] for pred in predictions]
            # print("prediction", predictions[:10], "target_to_dimension", prediction_target_uids)
            # print("predictions", len(predictions), predictions)
            # pred_labels = map_predictions_to_target_labels(predictions, target_to_dimension)
            print(f"[Normal Classifiacation]{len(set(pred_labels))} @@@@@@@ pred_labels[:30]:{pred_labels[:30]}")
        predictions = pred_labels

        # unique_label_list = list(set(labels))
        #  # Convert predictions to binary: non-zero becomes 1
        binary_predictions = [1 if pred != 0 else 0 for pred in predictions]
        binary_labels = [1 if label != 0 else 0 for label in labels]
        unique_label_list = list(set(labels))
        precision, recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)

        # Compute binary metrics
        binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(binary_labels, binary_predictions, average='binary', zero_division=0)
        binary_acc = accuracy_score(binary_labels, binary_predictions)

        return {
            "balanced_accuracy": balanced_acc,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "precision": precision,
            "recall": recall,
            "binary_accuracy": binary_acc,
            "binary_precision": binary_precision,
            "binary_recall": binary_recall,
            "binary_f1": binary_f1
        }
    
    def compute_test_metric(predictions, labels):
        # print("predictions",predictions)
        if not args.use_hierarchical_classifier:
            print(f"{args.loss_weight} - predictions: {type(predictions)}, labels: {type(labels),len(labels)}")
            print(f"{args.loss_weight} - predictions: {(predictions.shape)}, labels: {type(labels),len(labels)}")
           
        # unique_label_list = list(set(labels))
        #  # Convert predictions to binary: non-zero becomes 1
        binary_predictions = [1 if pred != 0 else 0 for pred in predictions]
        binary_labels = [1 if label != 0 else 0 for label in labels]
        unique_label_list = list(set(labels))
        precision, recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)

        # Compute binary metrics
        binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(binary_labels, binary_predictions, average='binary', zero_division=0)
        binary_acc = accuracy_score(binary_labels, binary_predictions)

        return {
            "balanced_accuracy": balanced_acc,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "precision": precision,
            "recall": recall,
            "binary_accuracy": binary_acc,
            "binary_precision": binary_precision,
            "binary_recall": binary_recall,
            "binary_f1": binary_f1
        }


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # Create a DataLoader for your test dataset
    # test_df = pd.read_csv(args.test_data_dir)
    # labels = list(test_df["assignedclass"])
    # tokenized_test_data = tokenizer(test_df['code'].tolist(), padding=True, truncation=True, return_tensors="pt")
    # # Create a custom dataset
    # test_dataset = vulDataset(tokenized_test_data,labels)
    # test_loader = DataLoader(test_dataset, batch_size=32)
    
    # trainer.train(resume_from_checkpoint = True)
    # Evaluate the model.
    def eval(model, tokenizer, args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Create a DataLoader for your test dataset
        test_df = pd.read_csv(args.test_data_dir)
        labels = list(test_df["assignedclass"])
        tokenized_test_data = tokenizer(test_df['code'].tolist(), padding=True, truncation=True, return_tensors="pt")
        # Create a custom dataset
        test_dataset = vulDataset(tokenized_test_data,labels)
        test_loader = DataLoader(test_dataset, batch_size=32)
    
        print("args.use_hierarchical_classifier",args.use_hierarchical_classifier, "args.use_focal_loss",args.use_focal_loss)
                    
        model.eval()
        prediction_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                if not args.use_hierarchical_classifier:
                    logits = logits.logits
                    logits = softmax(logits, dim=1)
                    print("logits", logits.shape)
                    predicted_labels = torch.argmax(logits, dim=1)
                    print("predicted_labels",predicted_labels)
                else:
                    predicted_labels = logits
                
                prediction_list.extend(predicted_labels.cpu().numpy().tolist())

        metrics = compute_metrics(prediction_list, labels)
        print(metrics)

    # print("@@@@@@@@@@@@@@@ Best Model with Best Acc:")
    # eval(acc_model)
    print("@@@@@@@@@@@@@@@ Best Model with Best macro_f1:")
    eval(model, tokenizer, args)
    # Walk through the output folder
    
    def get_checkpoint_folders(output_folder, metric_list):
        model_names = ['CodeBERT','GraphCodeBERT']
        loss_types = ['CE','FL','default','equalize','descendants','reachable_leaf_nodes']

        for model_name in model_names:
            if 'Graph' in args.checkpoint_dir:
                args.model_name = "microsoft/graphcodebert-base" 
            else:
                args.model_name = "microsoft/codebert-base" 

            for loss_type in loss_types:
                args.loss_weight = loss_type
                if loss_type == 'CE':
                    args.use_hierarchical_classifier = 'False'
                    args.use_focal_loss = 'False'
                elif loss_type == 'FL':
                    args.use_hierarchical_classifier = 'False'
                    args.use_focal_loss = 'True'
                else:
                    args.use_hierarchical_classifier = 'True'
                    
                key = f'{model_name}_{loss_type}'
                
                if key not in metric_list.keys():
                    metric_list[key] = {
                        'dir_name':[],
                        'metrics':[],}
                for root, dirs, files in os.walk(output_folder):
                    for dir_name in dirs:
                        print(f"dir_name:{dir_name}\nargs:{args}")
                        # Check if directory name contains 'CodeBERT' and 'default'
                        if model_name == dir_name[:len(model_name)] and loss_type in dir_name:
                            print("model_name == dir_name[:len(model_name)] and loss_type in dir_name",model_name == dir_name[:len(model_name)] , loss_type in dir_name)
                            full_path = os.path.join(root, dir_name)
                            args.checkpoint_dir = full_path
                            model, tokenizer = find_best_checkpoint_by_macro_f1(args)
                            metric = eval(model, tokenizer, args)
                            metric_list[key]['dir_name'].append(dir_name)
                            metric_list[key]['metrics'].append(metric)
                print(f"key:{key} | test metric:{metric_list[key]}")
        print("Final METRIC",metric_list)

    metric_list = {}
    output_folder = './outputs'
    # get_checkpoint_folders(output_folder, metric_list)
    
