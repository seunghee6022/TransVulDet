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
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score


def load_best_model_from_last_checkpoint(checkpoint_directory, args):
    # Find the last checkpoint folder
    last_checkpoint_folder = None
    for folder_name in sorted(os.listdir(checkpoint_directory), reverse=True):
        if folder_name.startswith("checkpoint-"):
            last_checkpoint_folder = folder_name
            break

    if last_checkpoint_folder is None:
        print("No checkpoint folders found.")
        return None

    # Construct the path to the trainer_state.json in the last checkpoint folder
    state_file_path = os.path.join(checkpoint_directory, last_checkpoint_folder, 'trainer_state.json')

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
    state_dict = torch.load(f'{best_model_path}/pytorch_model.bin', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    print(model)
    return model, tokenizer


checkpoint_directory = 'outputs/GraphCodeBERT_f1_reachable_leaf_nodes_afoRoeG9OEJcP5DI'
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
parser.add_argument('--num-trials', type=int, default=1, help='Number of trials for Optuna')
parser.add_argument('--use-bilstm', action='store_true', help='Flag for BiLSTM with Transformer Model')
parser.add_argument('--use-weight-sampling', action='store_true', help='Flag for using weight sampling')
parser.add_argument('--use-hierarchical-classifier', action='store_true', help='Flag for hierarchical classification') #--use-hierarchical-classifier --> true
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

# Parse the command line arguments
args = parser.parse_args()
args.loss_weight = 'reachable_leaf_nodes'
# print("args.use_hierarchical_classifier | args.use_bilstm",args.use_hierarchical_classifier, args.use_bilstm)
args.use_hierarchical_classifier = 'True'
# args.use_bilstm = 'False'
args.model_name = "microsoft/graphcodebert-base" 
# Initialize the tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
print("Args",args)
model, tokenizer = load_best_model_from_last_checkpoint(checkpoint_directory, args)


def compute_metric(predictions, labels):
    unique_label_list = list(set(labels))
    precision, recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0.0, labels=unique_label_list)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0.0, labels=unique_label_list)
    acc = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    return {
        "balanced_accuracy":balanced_acc,
        'accuracy': acc,
        'macro_f1':macro_f1,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Create a DataLoader for your test dataset
test_df = pd.read_csv('datasets_/test_dataset.csv')
labels = list(test_df["assignedclass"])
tokenized_test_data = tokenizer(test_df['code'].tolist(), padding=True, truncation=True, return_tensors="pt")
# Create a custom dataset
test_dataset = vulDataset(tokenized_test_data,labels)
test_loader = DataLoader(test_dataset, batch_size=32)


# trainer.train(resume_from_checkpoint = True)
# Evaluate the model.
model.eval()
prediction_list = []
with torch.no_grad():
    for batch in test_loader:
        loss, logits = model(**batch)
        # logits = outputs.logits
        predictions = softmax(logits, dim=1)
        predicted_labels = torch.argmax(predictions, dim=1)
        prediction_list.extend(predicted_labels.cpu().numpy().tolist())
        

metrics = compute_metric(prediction_list, labels)
print(metrics)