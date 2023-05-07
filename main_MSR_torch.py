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
from src.dataset import *
from src.models import *
from src.early_stopping import EarlyStopping

import random

# fix the seed to 42
random.seed(42)

# Load the CSV file as a Pandas DataFrame
MSR_df = pd.read_csv('data preprocessing/preprocessed datasets/MSR_labeled.csv')
MVD_df = pd.read_csv('data preprocessing/preprocessed datasets/MVD_labeled.csv')
print("MSR_df.columns\n",MSR_df.columns)
print("MVD_df.columns\n",MVD_df.columns)

MSR_df = MSR_df.sample(frac=0.001)
print(f'# of MSR sample dataset(0.1%): {MSR_df.shape[0]}')
MVD_df = MVD_df.sample(frac=0.001)
MVD_df = MVD_df.head(100)
print(f'# of MVD sample dataset(0.1%): {MVD_df.shape[0]}')


model_list = ['CodeBERT','BERT']

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
            #print("multi label, labels",labels.shape, labels)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, 1] if class_type == 'binary' else outputs.logits

        output_probs = F.softmax(logits, dim=0)
        #print("output_probs",output_probs.shape, output_probs)
        
        logit_pred_label = (output_probs >= 0.5).float().view(-1) if class_type == 'binary' else torch.argmax(output_probs, dim=1)
        #print("logit_pred_label", logit_pred_label)
        
        # compute the loss
        loss = criterion(logits, labels)
        
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
    precision /= total_num
    recall /= total_num
    f1 /= total_num
    
   
    return train_loss, train_acc, precision, recall, f1
    
def evaluate(model, class_type, average, device, criterion, val_loader):
    
    # Set the model to eval mode
    model.eval()
    
    # Evaluate on the validation set
    val_loss, val_acc, precision, recall, f1 = 0, 0, 0, 0, 0
    val_correct = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move the batch to device (e.g. GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if class_type == 'multi':
                labels = labels.argmax(dim=1)
                print("multi label, labels",labels.shape, labels)
  
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits[:, 1] if class_type == 'binary' else outputs.logits

            output_probs = F.softmax(logits, dim=0)
            #print("output_probs",output_probs.shape, output_probs)
            
            logit_pred_label = (output_probs >= 0.5).float().view(-1) if class_type == 'binary' else torch.argmax(output_probs, dim=1)
            #print("logit_pred_label", logit_pred_label)

            loss = criterion(logits, labels)
            
         
        val_loss += loss.item() * logits.size(0)
        val_correct += torch.sum(logit_pred_label == labels).item() 

        logit_pred_label_arr = logit_pred_label.cpu().detach().numpy().astype(int)
        target_label_arr = labels.cpu().detach().numpy().astype(int)

        print("logit_pred_label_arr", type(logit_pred_label_arr),logit_pred_label_arr)
        print("target_label_arr",type(target_label_arr),target_label_arr)
        
        precision, recall, f1, support = precision_recall_fscore_support(logit_pred_label_arr, target_label_arr ,average=average,zero_division=0)
        print("precision_recall_fscore_support:",precision, recall, f1, support)
        precision += precision
        recall += recall
        f1 += f1
   
    total_num = len(val_loader.dataset)
    val_loss /= total_num
    val_acc = val_correct / total_num
    precision /= total_num
    recall /= total_num
    f1 /= total_num
    
    return val_loss, val_acc, precision, recall, f1
    
def get_parameter_by_class_type(class_type):
    text_col_name = 'code'
    # get cwe label dictionary
    with open("data preprocessing/preprocessed datasets/total_cwe_dict.txt", "rb") as myFile:
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
        
    return text_col_name, label_col_name, num_labels, average, criterion
    
def train_classification(df, model_list, EPOCH, class_type, dataset_name):

    text_col_name, label_col_name, num_labels, average, criterion = get_parameter_by_class_type(class_type)
  
    # Split the dataset into training, validation, and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(get_texts(df[text_col_name]),get_labels(df[label_col_name],num_labels), test_size=0.2)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

    
    # Define the hyperparameters
    LEARNING_RATE = 1e-5
    EPSILON = 1e-5
    NUM_EPOCHS = EPOCH
    BATCH_SIZE = 8
  
    for model_name in model_list:
    
        print("--------------------Model name : ",model_name, "class_type",class_type,"criterion", criterion, "-------------------------")
        
        # make directory for each model to save logs, results, tokenizer and model
        output_dir = f'results/{dataset_name}_multi' if class_type == 'multi' else f'results/{dataset_name}_binary'
        output_dir = f'{output_dir}/{model_name}/Epoch{EPOCH}/batch{BATCH_SIZE}/early_stopping/'
        print(f"output_dir after adding model on the path: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"output dir is {output_dir}") if os.path.exists(output_dir) else print("Output dir is wrong!!!!!!!!!!!!!!!!!1")
        
        # Define the TensorBoard writer
        log_dir = f"{output_dir}/logs"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
         # Load the pre-trained model and tokenizer
        tokenizer, model = get_tokenizer_and_model(model_name, num_labels)
        
         # Move the model to the GPU device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Define the optimizer and loss function
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON, no_deprecation_warning=True)
        
        # Tokenize the input texts
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
        
        # Define the data loaders
        train_dataset = vulDataset(train_encodings, train_labels)
        val_dataset = vulDataset(val_encodings, val_labels)
        test_dataset = vulDataset(test_encodings, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
         # Define the scheduler
        total_steps = len(train_dataset) * NUM_EPOCHS // BATCH_SIZE
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
         
        early_stopping = EarlyStopping(patience=3, delta=0.001)
        
        train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
        for epoch in range(NUM_EPOCHS):
            # training loop
            train_loss, train_acc, train_precision, train_recall, train_f1_score = train(model, class_type, average, device, criterion, optimizer, scheduler, train_loader)

            # validation loop
            val_loss, val_acc, val_precision, val_recall, val_f1_score = evaluate(model, class_type, average, device, criterion, val_loader)
            
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            writer.add_scalar('Training loss', train_loss, epoch)
            writer.add_scalar('Training acc', train_acc, epoch)
            writer.add_scalar('Training f1', train_f1_score, epoch)
            # Log the model parameters and gradients
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)
                writer.add_histogram(name + "/grad", param.grad, epoch)
            
            writer.add_scalar('Validation loss', val_loss, epoch)
            writer.add_scalar('Validation acc', val_acc, epoch)
            writer.add_scalar('Validation precision', val_precision, epoch)
            writer.add_scalar('Validation recall', val_recall, epoch)
            writer.add_scalar('Validation f1', val_f1_score, epoch)
            
            # check for early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train F1 Score: {train_f1_score:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f},Val F1 Score: {val_f1_score:.3f}")


        # test result
        test_loss, test_acc, test_precision, test_recall, test_f1_score = evaluate(model, class_type, average, device, criterion, test_loader)
        
        writer.add_scalar('Test loss', test_loss)
        writer.add_scalar('Test acc', test_acc)
        writer.add_scalar('Test precision', test_precision)
        writer.add_scalar('Test recall', test_recall)
        writer.add_scalar('Test f1 score', test_f1_score)
            
        # Close the summary writer
        writer.close()
        
        # Print the evaluation metrics
        print(f"{model_name} Model result")
        print(f"Test set accuracy: {test_acc}")
        print(f"Test set f1 score: {test_f1_score}")
        print(f"Test set precision: {test_precision}")
        print(f"Test set recall: {test_recall}")
        print(f"Test set loss: {test_loss}")
        
        # Clear the figure
        plt.clf()
        
        # Plot the training and validation loss
        plt.plot(train_loss_list, label='Training Loss')
        plt.plot(val_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # Show the plot
        plt.show()
        # Save the plot to a PNG file
        plt.savefig(output_dir+'/Learning_curve_plot', format='png')

        # Clear the figure
        plt.clf()

        # Plot the training and validation accuracy
        plt.plot(train_acc_list, label='Training Accuracy')
        plt.plot(val_acc_list, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        # Show the plot
        plt.show()

        # Save the plot to a PNG file
        plt.savefig(output_dir+'/training_validation_acc_plot', format='png')

        # Save the trained model
        #torch.save(model.state_dict(), os.path.join(output_dir,f'{dataset_name}_model.bin'))
        #tokenizer.save_pretrained(output_dir)
        
        
        print("Training completed and the model is saved!")
        

EPOCH = 1
train_classification(MVD_df, model_list, EPOCH, 'binary', 'MVD')


