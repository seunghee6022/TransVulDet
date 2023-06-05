import os
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, f1_score, recall_score
from transformers import AdamW, get_linear_schedule_with_warmup

#import matplotlib.pyplot as plt
from src.dataset import get_labels, one_hot_to_labels, get_texts, vulDataset
from src.models import get_tokenizer_and_model
from src.early_stopping import EarlyStopping

from focal_loss.focal_loss import FocalLoss

import random

# fix the seed to 42
random.seed(42)

def train(model, class_type, average, device, criterion, optimizer, scheduler, train_loader):
   # Set the model to train mode
    model.train()
    
    cnt = 0
    # Train for one epoch
    train_loss = 0.0
    train_correct = 0
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
        '''
        # compute the loss
        loss = criterion(logits, labels)
        '''
        # calculate with focal loss
        m = torch.nn.Sigmoid() if class_type == 'binary' else torch.nn.Softmax(dim=-1)
        print(f"m(logits) shape: {m(logits).shape},\n m(logits): {m(logits)}")
        loss = criterion(m(logits), labels)
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        # Compute the accuracy
        train_loss += loss.item() * logits.size(0)
        train_correct += torch.sum(logit_pred_label == labels).item() 
        
        logit_pred_label_arr = logit_pred_label.cpu().detach().numpy().astype(int)
        target_label_arr = labels.cpu().detach().numpy().astype(int)
        
        precision, recall, f1, support = precision_recall_fscore_support(logit_pred_label_arr, target_label_arr ,average=average,zero_division=0)
        precision += precision
        recall += recall
        f1 += f1
   
    total_num = len(train_loader.dataset)
    train_loss /= total_num
    train_acc = train_correct / total_num
    precision /= len(train_loader)
    recall /= len(train_loader)
    f1 /= len(train_loader)
    
   
    return train_loss, train_acc, precision, recall, f1
    
def evaluate(model, class_type, average, device, criterion, val_loader):
    
    # Set the model to eval mode
    model.eval()
    
    # Evaluate on the validation set
    val_loss, val_acc, precision, recall, f1 = 0, 0, 0, 0, 0
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
            '''
            loss = criterion(logits, labels)
            '''
            # calculate with focal loss
            m = torch.nn.Sigmoid() if class_type == 'binary' else torch.nn.Softmax(dim=-1)
            print(f"m(logits) shape: {m(logits).shape},\n m(logits): {m(logits)}")
            loss = criterion(m(logits), labels)
            
            val_loss += loss.item() * logits.size(0)
            val_correct += torch.sum(logit_pred_label == labels).item() 
            
            total_logit_pred_label.append(labels.tolist())
            total_target_label.append(logit_pred_label.tolist())
            
            logit_pred_label_arr = logit_pred_label.cpu().detach().numpy().astype(int)
            target_label_arr = labels.cpu().detach().numpy().astype(int)

            print("logit_pred_label_arr", type(logit_pred_label_arr),logit_pred_label_arr)
            print("target_label_arr",type(target_label_arr),target_label_arr)
            
            precision, recall, f1, support = precision_recall_fscore_support(target_label_arr, logit_pred_label_arr ,average=average,zero_division=0)
            print("precision_recall_fscore_support:",precision, recall, f1, support)
            precision += precision
            recall += recall
            f1 += f1
   
        total_num = len(val_loader.dataset)
        val_loss /= total_num
        val_acc = val_correct / total_num
        precision /= len(val_loader)
        recall /= len(val_loader)
        f1 /= len(val_loader)
        
        '''
        total_target_label=  np.array(total_target_label)
        total_logit_pred_label=  np.array(total_logit_pred_label)
        print("input for the confusion matrix",total_target_label,total_logit_pred_label )
        # Assuming y_pred and y_true are your predicted and actual labels, respectively
        cm = confusion_matrix( total_target_label, total_logit_pred_label)
        print(cm)
        '''
    return val_loss, val_acc, precision, recall, f1
    
def get_parameter_by_class_type(class_type):
    print(f"{class_type} classification parameters")
    text_col_name = 'code'
    # get cwe label dictionary
    with open("data preprocessing/preprocessed datasets/total_cwe_dict.txt", "rb") as myFile:
        total_cwe_dict = pickle.load(myFile)

    labels = list(total_cwe_dict.keys())
    
    # 'vul' for binary, 'label' for multiclass
    if class_type == 'multi':
        label_col_name, num_labels, average = 'label', len(labels), 'weighted'
        #criterion = nn.CrossEntropyLoss()
        print("Total # of cwe ids: ",len(labels))

    else:
        label_col_name, num_labels, average = 'vul', 2, 'binary'
        #criterion= F.binary_cross_entropy_with_logits
        
    # Withoout class weights
    criterion = FocalLoss(gamma=0.7)

    return text_col_name, label_col_name, num_labels, average, criterion
    

# Define the objective function
def objective(trial, df, frac, class_type, model_name, dataset_name, weight_sampling=False):

    text_col_name, label_col_name, num_labels, average, criterion = get_parameter_by_class_type(class_type)
    print(text_col_name, label_col_name, num_labels, average, criterion)
  
    # Split the dataset into training, validation, and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(get_texts(df[text_col_name]),get_labels(df[label_col_name],num_labels), test_size=0.2)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

    # Define the search space for the hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay=trial.suggest_loguniform('weight_decay', 1e-3, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 2, 10)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    #warmup_step = trial.suggest_categorical('warmup_step', list(range(500, 1001, 100)))
    warmup_step = 500
    
    print(f"Selected Hyperparameters are:\n num_epochs: {num_epochs}\nlearning_rate:{learning_rate}\nbatch_size: {batch_size}\weight_decay: {weight_decay}\n dropout: {dropout}\nwarmup_step: {warmup_step}")

    # Build a Transformer model with the hyperparameters
    # Load the pre-trained model and tokenizer
    tokenizer, model = get_tokenizer_and_model(model_name, num_labels, dropout)
    
    # Move the model to the GPU device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, no_deprecation_warning=True)
    
    # Tokenize the input texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
    
    # Define the data loaders
    train_dataset = vulDataset(train_encodings, train_labels)
    val_dataset = vulDataset(val_encodings, val_labels)
    test_dataset = vulDataset(test_encodings, test_labels)
    
    if weight_sampling:

        # Calculate class weights
        train_labels_array = np.array(train_labels) if class_type=='binary' else one_hot_to_labels(train_labels)
        class_counts = np.bincount(train_labels_array)
        print("class_counts", len(class_counts),class_counts)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float) # inverse frequency
        print("class_weights", len(class_weights),class_weights)
        
        oversampling_factor = 10  # oversampling factor for class 0

        # Set the weight for class 0 to oversampling_factor times its original weight
        class_weights[0] *= oversampling_factor
        print("class_weights", len(class_weights),class_weights)

        # Create a weighted sampler
        weights = class_weights[train_labels_array]
        print(class_type,"weights",weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_steps = len(train_dataset) // batch_size * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_steps)
    
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    for epoch in range(num_epochs):
        # training loop
        train_loss, train_acc, train_precision, train_recall, train_f1_score = train(model, class_type, average, device, criterion, optimizer, scheduler, train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1 Score: {train_f1_score:.3f}")

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
    # validation 
    val_loss, val_acc, val_precision, val_recall, val_f1_score = evaluate(model, class_type, average, device, criterion, val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val F1 Score: {val_f1_score:.3f}")   
    
    return val_acc
        
if __name__ == "__main__":
    # fix the seed to 42
    random.seed(42)

    # Load the CSV file as a Pandas DataFrame
    MSR_df = pd.read_csv('data preprocessing/preprocessed datasets/MSR_labeled.csv')
    MVD_df = pd.read_csv('data preprocessing/preprocessed datasets/MVD_labeled.csv')

    # model_lists = ['BERT','CodeBERT','CodeRoBERTa','CodeBERTa' ,'T5','CodeT5', 'GPT','CodeGPT' ]
    
    model_name = 'BERT' 
    dataset_name = 'MSR'
    class_type = 'binary'
    n_trials = 10
    frac = 1
    
    MSR_df = MSR_df.sample(frac=frac)
    MVD_df = MVD_df.sample(frac=frac)

    df = MSR_df if dataset_name == 'MSR' else MVD_df
    
    print(f"# of {dataset_name} sample dataset({frac}): {df.shape[0]}")
    print("columns\n",df.columns)
 
    # Set up the study
    study = optuna.create_study(direction='maximize')

    # Run the optimization
    study.optimize(lambda trial: objective(trial, df, frac, class_type, model_name, dataset_name, False), n_trials=n_trials)

    # Create a new directory
    output_dir = f'results/{dataset_name}_multi' if class_type == 'multi' else f'results/{dataset_name}_binary'
    output_dir = f'{output_dir}/Optuna/{model_name}/n_trials{n_trials}/{frac*100}%'
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
    
    print("Hyperparameter tuning is done!!!")




