import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import optuna

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, f1_score, recall_score, confusion_matrix, balanced_accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup

#import matplotlib.pyplot as plt
from src.dataset import vulDataset, get_labels, get_texts, one_hot_to_labels
from src.models import get_tokenizer_and_model
from src.early_stopping import EarlyStopping

import random

from focal_loss.focal_loss import FocalLoss


def train(model, class_type, average, device, criterion, optimizer, scheduler, train_loader, step_size=1000):
   # Set the model to train mode
    model.train()

    step = 0
    train_loss_list, train_acc_list, train_recall_list, train_precision_list, train_f1_list = [], [], [], [], []
    while step < step_size:
        # Train for one epoch
        train_loss_list, train_acc_list, train_bal_acc_list, train_recall_list, train_precision_list, train_f1_list = [], [], [], [], [], []
        for batch in train_loader:
            # Move the batch to device (e.g. GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) 
            if class_type == 'multi':
                labels = labels.argmax(dim=1)
            #labels = labels.to(torch.int) 
            print("Train - labels",labels.shape, labels)  
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, 1] if class_type == 'binary' else outputs.logits

            output_probs = F.softmax(logits, dim=0)
            #print("output_probs",output_probs.shape, output_probs)
            
            logit_pred_label = (output_probs >= 0.5).float().view(-1) if class_type == 'binary' else torch.argmax(output_probs, dim=1)
            #print("logit_pred_label", logit_pred_label)
            
            # compute the loss
            #loss = criterion(logits, labels)
            
            # calculate with focal loss
            m = torch.nn.Sigmoid() if class_type == 'binary' else torch.nn.Softmax(dim=-1)
            print(f"m(logits) shape: {m(logits).shape},\n m(logits): {m(logits)}")
            loss = criterion(m(logits), labels.long())
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            # Compute the accuracy
            train_loss = loss.item() * logits.size(0)
            train_correct = torch.sum(logit_pred_label == labels).item() 
            
            logit_pred_label_arr = logit_pred_label.cpu().detach().numpy().astype(int)
            target_label_arr = labels.cpu().detach().numpy().astype(int)
            
            train_precision, train_recall, train_f1, support = precision_recall_fscore_support(logit_pred_label_arr, target_label_arr ,average=average,zero_division=0)
            train_acc = train_correct/len(batch)
            train_bal_acc = balanced_accuracy_score(logit_pred_label_arr, target_label_arr)
    
    
            # Update step count
            step += 1

            if step % 100 == 0 :
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                train_bal_acc_list.append(train_bal_acc)
                train_recall_list.append(train_recall)
                train_precision_list.append(train_precision) 
                train_f1_list.append(train_f1)
                print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train Balanced Acc : {train_bal_acc:.3f} Train F1 Score: {train_f1:.3f}, Train Precision: {train_precision:.3f}, Train Recall: {train_recall:.3f}")
    

            # Break out of the loop if reached the desired number of steps
            if step >= step_size:
                break
    
        # Break out of the outer while loop if reached the desired number of steps
        if step >= step_size:
            break

    return train_loss_list, train_acc_list, train_bal_acc_list, train_precision_list, train_recall_list, train_f1_list
    
    
def evaluate(model, class_type, average, device, criterion, val_loader):
    
    # Set the model to eval mode
    model.eval()
    
    # Evaluate on the validation set
    val_loss, val_acc, val_bal_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0, 0 
    val_correct = 0
    
    total_target_label, total_logit_pred_label = [], []

    with torch.no_grad():
        for batch in val_loader:
            # Move the batch to device (e.g. GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if class_type == 'multi':
                labels = labels.argmax(dim=1)
            #labels = labels.to(torch.int)
  
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits[:, 1] if class_type == 'binary' else outputs.logits

            output_probs = F.softmax(logits, dim=0)
            #print("output_probs",output_probs.shape, output_probs)
            
            logit_pred_label = (output_probs >= 0.5).float().view(-1) if class_type == 'binary' else torch.argmax(output_probs, dim=1)
            #print("logit_pred_label", logit_pred_label)

            #loss = criterion(logits, labels)
            
            # calculate with focal loss
            m = torch.nn.Sigmoid() if class_type == 'binary' else torch.nn.Softmax(dim=-1)
            print(f"m(logits) shape: {m(logits).shape},\n m(logits): {m(logits)}")
            loss = criterion(m(logits), labels.long())
            
            val_loss += loss.item() * logits.size(0)
            val_correct += torch.sum(logit_pred_label == labels).item() 
            
            total_logit_pred_label.append(labels.tolist())
            total_target_label.append(logit_pred_label.tolist())
            
            logit_pred_label_arr = logit_pred_label.cpu().detach().numpy().astype(int)
            target_label_arr = labels.cpu().detach().numpy().astype(int)

            print("logit_pred_label_arr", type(logit_pred_label_arr),logit_pred_label_arr)
            print("target_label_arr",type(target_label_arr),target_label_arr)
            
            precision, recall, f1, support = precision_recall_fscore_support(target_label_arr, logit_pred_label_arr ,average=average,zero_division=0)
            bal_acc = balanced_accuracy_score(logit_pred_label_arr, target_label_arr )
            
            print("precision_recall_fscore_support:",precision, recall, f1, support)
            val_precision += precision
            val_recall += recall
            val_f1 += f1
            val_bal_acc += bal_acc
   
        total_num = len(val_loader.dataset)
        val_loss /= total_num
        val_acc = val_correct / total_num
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        val_bal_acc /= len(val_loader)

    return val_loss, val_acc, val_bal_acc, val_precision, val_recall, val_f1
    
def get_parameter_by_class_type(class_type):
    print(f"{class_type} classification parameters")
    text_col_name = 'code'
    # get cwe label dictionary
    with open("data_preprocessing/preprocessed_datasets/total_cwe_dict.txt", "rb") as myFile:
        total_cwe_dict = pickle.load(myFile)

    labels = list(total_cwe_dict.keys())
    
    # 'vul' for binary, 'label' for multiclass
    if class_type == 'multi':
        label_col_name, num_labels, average = 'label', len(labels), 'weighted'
        criterion = nn.CrossEntropyLoss()
        print("Total # of cwe ids: ",len(labels))

    else:
        label_col_name, num_labels, average = 'vul', 2, 'binary'
        criterion= F.binary_cross_entropy_with_logits
        
    # Withoout class weights
    criterion = FocalLoss(gamma=0.7)

    return text_col_name, label_col_name, num_labels, average, criterion

# Define a function to update the lists and plot the results
def update_lists_and_plot(params, output_dir, train_losses, val_loss, train_accs, val_acc, train_precisions, val_precision, train_recalls, val_recall, train_f1s, val_f1):
  
    # Plot the train and validation losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    # Plot validation loss as a straight baseline
    plt.axhline(y=val_loss, color='r', linestyle='--', label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_plot_{str(params)}.png')
    
    # Plot the train and validation balanced accuracies
    plt.figure()
    plt.plot(train_bal_accs, label='Train Balanced Accuracy')
    plt.axhline(val_bal_acc, color='r', linestyle='--', label='Validation Balanced Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.savefig(f'{output_dir}/balanced_accuracy_plot_{str(params)}.png')
    

    # Plot the train and validation accuracies
    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.axhline(val_acc, color='r', linestyle='--', label='Validation Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{output_dir}/accuracy_plot_{str(params)}.png')

    # Plot the train and validation precision, recall, and F1 score
    plt.figure()
    plt.plot(train_precisions, label='Train Precision')
    plt.axhline(val_precision, color='r', linestyle='--', label='Validation Precision')
    plt.plot(train_recalls, label='Train Recall')
    plt.axhline(val_recall, color='g', linestyle='--', label='Validation Recall')
    plt.plot(train_f1s, label='Train F1 Score')
    plt.axhline(val_f1, color='b', linestyle='--', label='Validation F1 Score')
    plt.xlabel('Step')
    plt.ylabel('Score ')
    plt.legend()
    plt.savefig(f'{output_dir}/score_plot_{str(params)}.png')

def get_oversampled_dataset(df, tokenizer, text_col_name, label_col_name, class_type, num_labels):
    
    # Split the dataset into training, validation, and test sets
    texts, labels = get_texts(df[text_col_name]), get_labels(df[label_col_name], num_labels)
  
    # Create the RandomOverSampler object - minority for binary and none is for multiclass
    oversampler = RandomOverSampler(sampling_strategy="minority", random_state=42) if class_type=="binary" else RandomOverSampler(random_state=42)
    
    if class_type == 'multi':
        # Convert one-hot encoded labels back to original format
        labels = one_hot_to_labels(labels)

    # Keep the same number of samples in texts and labels
    min_samples = min(len(texts), len(labels))
    texts = texts[:min_samples]
    labels = labels[:min_samples]

    # Perform the oversampling (array type)
    texts = np.array(texts).reshape(-1, 1)
    resampled_texts, resampled_labels = oversampler.fit_resample(texts, labels)
    
    # Tokenize the input texts
    resampled_encodings = tokenizer(resampled_texts, truncation=True, padding=True, return_tensors='pt')
    print(f"resampled_encodings shape {resampled_encodings['input_ids'].shape}  type {type(resampled_encodings)}")

    # Create a vulDataset with the oversampled data - multiclass label should be one-hot-encoded
    if class_type == 'multi':
        # convert index label to one-hot-encoded labels
        resampled_one_hot_labels = torch.eye(num_labels)[resampled_labels]
        resampled_labels = resampled_one_hot_labels

    oversampled_dataset = vulDataset(resampled_encodings, resampled_labels)
    # Print the length of the dataset
    dataset_length = len(oversampled_dataset)
    print("Length of the dataset:", dataset_length)

    return oversampled_dataset
    
# Define the objective function
def objective(trial, train_df, val_df, test_df, output_dir, class_type, model_name, step_size, weight_sampling=False):

    text_col_name, label_col_name, num_labels, average, criterion = get_parameter_by_class_type(class_type)
    print(text_col_name, label_col_name, num_labels, average, criterion)

    # Define the search space for the hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    batch_size = 4
    #warmup_step = trial.suggest_categorical('warmup_step', list(range(500, 1001, 100)))
    weight_decay=trial.suggest_loguniform('weight_decay', 1e-3, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    warmup_step = 500
 
    params = {"learning_rate":learning_rate, "batch_size":batch_size,"weight_decay":weight_decay, "dropout":dropout, "warmup_step":warmup_step}
    print(f"Selected Hyperparameters are:\n {str(params)}")

    # Build a Transformer model with the hyperparameters
    # Load the pre-trained model and tokenizer
    tokenizer, model = get_tokenizer_and_model(model_name, num_labels, dropout)
    
    # Move the model to the GPU device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, no_deprecation_warning=True)
    
    # Define the dataset and dataloader
    train_dataset = get_oversampled_dataset(train_df, tokenizer, text_col_name, label_col_name, class_type, num_labels)
    val_dataset = get_oversampled_dataset(val_df, tokenizer, text_col_name, label_col_name, class_type, num_labels)
    test_dataset = get_oversampled_dataset(test_df, tokenizer, text_col_name, label_col_name, class_type, num_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=step_size)

    train_losses, train_accs, train_bal_accs, train_precisions, train_recalls, train_f1s = train(model, class_type, average, device, criterion, optimizer, scheduler, train_loader, step_size)
    
    # validation 
    val_loss, val_acc, val_bal_acc, val_precision, val_recall, val_f1 = evaluate(model, class_type, average, device, criterion, val_loader)
    print(f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val Balanced Acc: {val_bal_acc:.3f}, Val F1 Score: {val_f1:.3f} Val Precision: {val_precision:.3f}, Val Recall: {val_recall:.3f}") 
    
    update_lists_and_plot(params, output_dir, train_losses, val_loss, train_accs, val_acc, train_bal_accs, val_bal_acc, train_precisions, val_precision, train_recalls, val_recall, train_f1s, val_f1)
  
    return val_bal_acc
        
if __name__ == "__main__":
    # fix the seed to 42
    random.seed(42)

    # Load the CSV file as a Pandas DataFrame
    train_df = pd.read_csv('data_preprocessing/preprocessed_datasets/train_data.csv')
    val_df = pd.read_csv('data_preprocessing/preprocessed_datasets/val_data.csv')
    test_df = pd.read_csv('data_preprocessing/preprocessed_datasets/test_data.csv')

    train_df = train_df.sample(0.3)
    val_df = val_df.sample(0.3)
    test_df = test_df.sample(0.3)

    print(f"# of rows in train_df dataset: {train_df.shape[0]}")
    print(f"# of rows in val_df dataset: {val_df.shape[0]}")
    print(f"# of rows in test_df dataset: {test_df.shape[0]}")
    print("columns\n",train_df.columns)

    model_name = 'BERT' 
    class_type = 'multi'
    step_size = 100 if class_type == 'multi' else 100
    n_trials = 3
    
    # Create a new directory
    output_dir = f'results/multi' if class_type == 'multi' else f'results/binary'
    output_dir = f'{output_dir}/HTO/{model_name}/step_size{step_size}/n_trials{n_trials}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the study
    study = optuna.create_study(direction='maximize')

    # Run the optimization -
    study.optimize(lambda trial: objective(trial, train_df, val_df, test_df, output_dir, class_type, model_name, step_size), n_trials=n_trials)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Retrieve the best hyperparameters
    best_params = study.best_params
    print(f"Best params: {best_params}")
    with open(f'{output_dir}/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    best_value = study.best_value
    print(f"best_value: {best_value}")
    with open(f'{output_dir}/best_value.txt', 'w') as f:
        f.write(str(best_value))
        
    trials_dataframe = study.trials_dataframe()
    print(f"trials_dataframe: {trials_dataframe}")
    trials_dataframe.to_csv(f'{output_dir}/trials.csv', index=False)

    print(f"{model_name} with concatenated_df(frac={frac}) with {n_trials} trials, step_size {step_size} on {class_type} classification \nHyperparameter tuning is done!!!")
    
 

