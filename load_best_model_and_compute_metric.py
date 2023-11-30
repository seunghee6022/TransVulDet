import os
import torch
import json
from datasets import load_dataset
from transformers import AutoModel
from src.graph import create_graph_from_json
from src.classifier_debug import get_model_and_tokenizer
from src.dataset import vulDataset
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import subprocess

def load_best_checkpoint_by_macro_f1(args):
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
                
    best_model_path = f'{args.checkpoint_dir}/{last_checkpoint_folder}'
  
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
    parser = argparse.ArgumentParser(description="Fine-tuning Models")

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
    parser.add_argument('--output-dir', type=str, default='./outputs', help='HPO output directory')
    parser.add_argument('--logging-dir', type=str, default='./logs', help='Trainer log directory')
    parser.add_argument('--eval-metric', type=str, default='f1', help='Evaluation metric')
   
    # Parse the command line arguments
    args = parser.parse_args()
  
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
    target_to_dimension = {target:idx for idx,target in enumerate(prediction_target_uids)}
    graph = create_graph_from_json(paths_dict_data, max_depth=None)

    model, tokenizer = load_best_checkpoint_by_macro_f1(args)
   
    def compute_metrics(predictions, labels):
        labels = mapping_cwe_to_target_label(labels, target_to_dimension)
     
        if args.use_hierarchical_classifier:
            pred_dist = model.deembed_dist(predictions) # get probabilities of each nodes
            pred_cwe_labels = model.dist_to_cwe_ids(pred_dist)
            pred_labels = mapping_cwe_to_target_label(pred_cwe_labels, target_to_dimension)
        else:
            pred_labels = predictions
        predictions = pred_labels

        # Convert predictions to binary: non-zero becomes 1
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

    def eval(model, tokenizer, args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_df = pd.read_csv(args.test_data_dir)
        labels = list(test_df["assignedclass"])
        tokenized_test_data = tokenizer(test_df['code'].tolist(), padding=True, truncation=True, return_tensors="pt")
        # Create a custom dataset
        test_dataset = vulDataset(tokenized_test_data,labels)
        test_loader = DataLoader(test_dataset, batch_size=32)
              
        model.eval()
        prediction_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                if not args.use_hierarchical_classifier:
                    logits = logits.logits
                    logits = softmax(logits, dim=1)
                    predicted_labels = torch.argmax(logits, dim=1)
                else:
                    predicted_labels = logits
                
                prediction_list.extend(predicted_labels.cpu().numpy().tolist())

        metrics = compute_metrics(prediction_list, labels)
        print(metrics)

    eval(model, tokenizer, args)
    