import os
import torch
from torch.utils.data import DataLoader, TensorDataset
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

# Load the CSV file as a Pandas DataFrame
MSR_df = pd.read_csv('data preprocessing/preprocessed datasets/MSR_labeled.csv')
print(MSR_df.columns)

#df = MSR_df.head(100)
df = MSR_df


# model_lists = ['BERT','CodeBERT','CodeRoBERTa','CodeBERTa' ,'T5','CodeT5', 'GPT','CodeGPT' ]
model_list = ['CodeBERT','BERT']


def train_classification(df, model_list, EPOCH, class_type):
    text_col_name = 'code'

    # 'vul' for binary, 'label' for multiclass
    if class_type == 'multi':
        label_col_name = 'label'
        output_dir = './results/MSR_multi'
        # get cwe label dictionary
        with open("data preprocessing/preprocessed datasets/total_cwe_dict.txt", "rb") as myFile:
            total_cwe_dict = pickle.load(myFile)

        labels = list(total_cwe_dict.keys())
        print("Total # of cwe ids: ",len(labels))

        num_labels = len(labels)
        average ='weighted'
        criterion = nn.CrossEntropyLoss()
        
        print("MSR MULTI CLASSIFICATION TRAINING START-------------------------------------------------------")

    else:
        label_col_name = 'vul'
        output_dir = './results/MSR_binary'
        num_labels = 2
        average = 'binary'
        criterion= F.binary_cross_entropy_with_logits
        
        print("MSR BINARY CLASSIFICATION TRAINING START-------------------------------------------------------")
    

    # Split the dataset into training, validation, and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(get_texts(df[text_col_name]),get_labels(df[label_col_name],num_labels), test_size=0.2)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

    # Load the pre-trained model and tokenizer
  
    for model_name in model_list:
        print("--------------------Model name : ",model_name, "class_type",class_type,"criterion", criterion, "-------------------------")
        tokenizer, model = get_tokenizer_and_model(model_name, num_labels)
        
         # Move the model to the GPU device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Define the hyperparameters
        LEARNING_RATE = 1e-5
        EPSILON = 1e-5
        NUM_EPOCHS = EPOCH
        BATCH_SIZE = 32

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


        # Define the TensorBoard writer
        writer = SummaryWriter()
        
        
        # Train the model
        train_loss_list, val_loss_list, train_acc_list, val_acc_list = [],[],[],[]
        for epoch in range(NUM_EPOCHS):
        
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

                # Log metrics
                batch_loss = loss.item()
                print(f"Epoch {epoch}, batch loss: {batch_loss}")
                
                # Compute the accuracy
                train_loss += loss.item() * logits.size(0)
                train_correct += torch.sum(logit_pred_label == labels).item() 
            
            
            train_loss /= len(train_dataset)
            train_acc = train_correct / len(train_dataset)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)

                
            # Set the model to eval mode
            model.eval()
            
            # Evaluate on the validation set
            val_loss = 0.0
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
                print(f"{epoch} val_correct {val_correct}" )  
            
            val_loss /= len(val_dataset)
            val_acc = val_correct / len(val_dataset)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            # Print the validation metrics
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, NUM_EPOCHS, train_loss, train_acc, val_loss, val_acc))
            
        # Evaluate the model on the test set
        print("Evaluating the model on the test set...")
        model.eval()
        with torch.no_grad():
            test_loss, test_accuracy, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
            for batch in test_loader:
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

                loss = criterion(logits, labels)
              
                test_loss += loss.item()
                print("# of logit_pred_label == target_label: ",torch.sum(logit_pred_label ==  labels).item() )
                test_accuracy += torch.sum(logit_pred_label ==  labels).item() 
                
                logit_pred_label_arr = logit_pred_label.cpu().detach().numpy().astype(int)
                target_label_arr = labels.cpu().detach().numpy().astype(int)

                print("logit_pred_label_arr", type(logit_pred_label_arr),logit_pred_label_arr)
                print("target_label_arr",type(target_label_arr),target_label_arr)
                
                test_precision, test_recall, test_f1, support = precision_recall_fscore_support(logit_pred_label_arr, target_label_arr ,average=average,zero_division=0)
                print("precision_recall_fscore_support:",test_precision, test_recall, test_f1, support)
                test_precision += test_precision
                test_recall += test_recall
                test_f1 += test_f1
                
            print("len(test_loader)",len(test_loader))
            test_loss /= len(test_loader)
            test_accuracy /= len(test_dataset)
            test_precision /= len(test_loader)
            test_recall /= len(test_loader)
            test_f1 /= len(test_loader)

        
        # Print the evaluation metrics
        print(f"{model_name} Model result")
        print(f"Test set accuracy: {test_accuracy}")
        print(f"Test set f1 score: {test_f1}")
        print(f"Test set precision: {test_precision}")
        print(f"Test set recall: {test_recall}")
        print(f"Test set loss: {test_loss}")
        
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
        plt.savefig(output_dir+'Learning_curve_plot_{model_name}_{class_type}_E{NUM_EPOCHS}', format='png')

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
        plt.savefig(output_dir+f'training_validation_acc_plot_{model_name}_{class_type}_E{NUM_EPOCHS}', format='png')

        # Save the trained model
        #torch.save(model.state_dict(), os.path.join(output_dir,f'pytorch_{model_name}_MSR100_{class_type}_E{NUM_EPOCHS}.bin'))
        #tokenizer.save_pretrained(output_dir)
        
        
        print("Training completed and the model is saved!")
        

EPOCH = 3
train_classification(df, model_list, EPOCH, 'multi')


