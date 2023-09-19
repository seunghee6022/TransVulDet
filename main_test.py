import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import optuna

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
from transformers import AdamW

#import matplotlib.pyplot as plt
from src.dataset import OversampledDatasetGenerator, vulDataset, get_texts, get_labels
from src.models import get_tokenizer_and_model

import random
# from focal_loss.focal_loss import FocalLoss
from focal_loss import FocalLoss


def train(model, class_type, average, device, criterion, optimizer, train_dataloader, writer,step_size=1000):
   # Set the model to train mode
    model.train()

    global_step = 0
    total_num, train_correct, train_loss= 0, 0, 0.0
    train_loss_list, train_acc_list, train_bal_acc_list, train_recall_list, train_precision_list, train_f1_list = [], [], [], [], [], []
    while global_step < step_size:
        # Train for one epoch
        train_correct, train_loss, train_precision, train_recall, train_f1 = 0, 0.0, 0, 0, 0
        for batch in train_dataloader:
            print(f" TRAIN START ++++++++++++++++++++ BATCH | Global step {global_step} ++++++++++++++++++++++++")
            print("Batch",batch)
            len_batch = len(batch[1])
            print(f"len of batch:{len_batch}\nbatch:{batch}")
            inputs = {'input_ids': batch[0]['input_ids'].to(device),
                    'attention_mask': batch[0]['attention_mask'].to(device),
                    }
            
            if class_type == 'multi':
                labels = batch[1].argmax(dim=1).to(device)
            else:
                labels = torch.tensor(batch[1]).to(device)
        
            outputs = model(**inputs)
            logits = outputs.logits
            print("logits",logits)
            output_probs = F.softmax(logits, dim=1)
        

            logit_pred_label = (output_probs[:,1] >= 0.5).float().view(-1) if class_type == 'binary' else torch.argmax(output_probs, dim=1)
            # calculate with focal loss
            # m = torch.nn.Sigmoid() if class_type == 'binary' else torch.nn.Softmax(dim=1)
            print(f"{class_type} | output_probs = {output_probs}")
            # print(f"{class_type} | m(logits) = {m(logits)}")
            print(f"{class_type} | labels = {labels.long()}")
            loss = criterion(output_probs, labels.long())
            print(f"{class_type} | Training loss = {loss.item()}")
            # Backward pass
            optimizer.zero_grad() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Compute the accuracy
            logit_pred_label_arr = logit_pred_label.cpu().detach().numpy().astype(int)
            target_label_arr = labels.cpu().detach().numpy().astype(int)
            print("logit_pred_label_arr", type(logit_pred_label_arr),logit_pred_label_arr)
            print("target_label_arr",type(target_label_arr),target_label_arr)
            
            train_precision, train_recall, train_f1, support = precision_recall_fscore_support(logit_pred_label_arr, target_label_arr ,average=average,zero_division=0)

            total_num += len_batch
            train_correct += torch.sum(logit_pred_label == labels).item()
            train_loss += loss.item()
            train_acc = train_correct / total_num
            train_bal_acc = balanced_accuracy_score(logit_pred_label_arr, target_label_arr)
            global_step += 1  # increment the global step count
          
            if global_step % 10 == 0:  # adjust this or use batch_idx variable according to your needs
                print(f'Loss/Acc after mini-batch(global_step: {global_step}): {train_loss/(global_step)}/{train_acc}')
                writer.add_scalar('Training Loss', train_loss/(global_step), global_step)
                writer.add_scalar('Training Acc', train_acc, global_step)
                writer.add_scalar('Training Balanced Acc', train_bal_acc, global_step)
                writer.add_scalar('Training F1', train_f1, global_step)
                writer.add_scalar('Training Recall', train_recall, global_step)
                writer.add_scalar('Training Precision', train_precision, global_step)
            
                train_loss_list.append(train_loss/global_step)
                train_acc_list.append(train_acc)
                train_bal_acc_list.append(train_bal_acc)
                train_recall_list.append(train_recall)
                train_precision_list.append(train_precision) 
                train_f1_list.append(train_f1)
                print(f"Train Loss: {loss.item():.3f}, Train Acc: {train_acc:.3f}, Train Balanced Acc : {train_bal_acc:.3f} Train F1 Score: {train_f1:.3f}, Train Precision: {train_precision:.3f}, Train Recall: {train_recall:.3f}")
        
                # Update step count
                global_step += 1

                # Break out of the loop if reached the desired number of steps
                if global_step >= step_size:
                    break
        
            # Break out of the outer while loop if reached the desired number of steps
            if global_step >= step_size:
                break

    return model, train_loss_list, train_acc_list, train_bal_acc_list, train_precision_list, train_recall_list, train_f1_list
    
def evaluate(model, class_type, average, device, criterion, val_dataloader, writer):
    
    # Set the model to eval mode
    model.eval()
    
    global_step = 0  # add a global step count
    total_logit_pred_label, total_target_label = [], []
    val_loss, val_acc, val_bal_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0, 0 
    val_correct = 0
    total_num = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            print(f" EVAL START ++++++++++++++++++++ BATCH {idx+1} | Global step {global_step} ++++++++++++++++++++++++")
            print("Batch",batch)
            len_batch = len(batch[1])
            inputs = {'input_ids': batch[0]['input_ids'].to(device),
                      'attention_mask': batch[0]['attention_mask'].to(device)}
        
            if class_type == 'multi':
                labels = batch[1].argmax(dim=1).to(device)
            else:
                labels = torch.tensor(batch[1]).to(device)
            print("inputs", inputs)
            outputs = model(**inputs)
            print("outputs", outputs)
            logits = outputs.logits
            print("logits",logits)
            output_probs = F.softmax(logits, dim=1)
        
            print(f"{class_type} | output_probs = {output_probs}")
            print(f"{class_type} | labels = {labels.long()}")
            loss = criterion(output_probs, labels.long())
            print(f"{class_type} | Val loss = {loss.item()}")


            val_loss += loss.item() 
            logit_pred_label = (output_probs[:,1] >= 0.5).float().view(-1) if class_type == 'binary' else torch.argmax(output_probs, dim=1)
            val_correct += torch.sum(logit_pred_label == labels).item() 
            
            total_logit_pred_label.append(labels.tolist())
            total_target_label.append(logit_pred_label.tolist())
            
            logit_pred_label_arr = logit_pred_label.cpu().detach().numpy().astype(int)
            target_label_arr = labels.cpu().detach().numpy().astype(int)

            print("logit_pred_label_arr", type(logit_pred_label_arr),logit_pred_label_arr)
            print("target_label_arr",type(target_label_arr),target_label_arr)
            
            precision, recall, f1, _ = precision_recall_fscore_support(target_label_arr, logit_pred_label_arr ,average=average,zero_division=0)
            bal_acc = balanced_accuracy_score(logit_pred_label_arr, target_label_arr )
            print("precision_recall_fscore_support:",precision, recall, f1)
            val_precision += precision
            val_recall += recall
            val_f1 += f1
            val_bal_acc += bal_acc
            total_num+=len_batch

            if global_step % 10 == 0:  # adjust this or use batch_idx variable according to your needs
                print(f'Eval Loss|ACC|Bal Acc after mini-batch {global_step}: {loss.item()}|{val_correct/total_num}|{bal_acc}')
                writer.add_scalar('Validation Loss', val_loss/(idx+1), global_step)
                writer.add_scalar('Validation Acc', val_correct/total_num, global_step)
                writer.add_scalar('Validation Balanced Acc', val_bal_acc/(idx+1), global_step)
                writer.add_scalar('Validation F1', val_f1/(idx+1), global_step)

            global_step += 1  # increment the global step count

        val_loss /= len(val_dataloader)
        val_acc = val_correct / total_num
        val_precision /= len(val_dataloader)
        val_recall /= len(val_dataloader)
        val_f1 /= len(val_dataloader)
        val_bal_acc /= len(val_dataloader)
        print(f'Eval Loss|ACC|Bal Acc|f1 after evaluation: {val_loss}|{val_acc}|{val_bal_acc}|{val_f1}')

        # if class_type == 'binary':
        #     total_target_label=  np.array(total_target_label).astype(int)
        #     total_logit_pred_label=  np.array(total_logit_pred_label).astype(int)
        #     print("input for the confusion matrix",total_target_label,total_logit_pred_label )
        #     # Assuming y_pred and y_true are your predicted and actual labels, respectively
        #     cm = confusion_matrix( total_target_label, total_logit_pred_label)
        #     print(cm)
        
    return val_loss, val_acc, val_bal_acc, val_precision, val_recall, val_f1
    
def get_parameter_by_class_type(class_type):
    print(f"{class_type} classification parameters")
    text_col_name = 'code'
    # get cwe label dictionary
    print(os.getcwd())
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
def update_lists_and_plot(params, output_dir, train_losses, val_loss, train_accs, val_acc, train_bal_accs, val_bal_acc, train_precisions, val_precision, train_recalls, val_recall, train_f1s, val_f1):
  
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


# Define the objective function
def objective(trial, train_df, val_df, test_df, output_dir, class_type, model_name):

    text_col_name, label_col_name, num_labels, average, criterion = get_parameter_by_class_type(class_type)
    print(text_col_name, label_col_name, num_labels, average, criterion)
  
    # Define the search space for the hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    batch_size = 8
    # batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
    step_size = trial.suggest_categorical('step_size', list(range(10, 101, 10)))
    weight_decay=trial.suggest_float('weight_decay', 1e-3, 1e-2)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    warmup_step = 0
    # warmup_step = trial.suggest_categorical('warmup_step', list(range(0, 101, 10)))
 
    print("HPO Start - Eval metric is Eval loss!!!!!")
    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% {model_name} {class_type} dataset ({frac}) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    params = {"learning_rate":learning_rate, "batch_size":batch_size,"weight_decay":weight_decay, "dropout":dropout, "step_size":step_size}
    print(f"Selected Hyperparameters are:\n {str(params)}")
    
    writer = SummaryWriter('runs')
    # Load the pre-trained model and tokenizer
    tokenizer, model = get_tokenizer_and_model(model_name, num_labels, dropout)
    
    # Move the model to the GPU device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define the optimizer and loss function
    criterion = FocalLoss(gamma=0.7)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, no_deprecation_warning=True)
    
    val_texts, val_labels = get_texts(val_df[text_col_name]), get_labels(val_df[label_col_name],num_labels)
    test_texts, test_labels = get_texts(test_df[text_col_name]), get_labels(test_df[label_col_name],num_labels)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

    print("OversampledDatasetGenerator - Train START")
    train_dataset = OversampledDatasetGenerator(train_df, tokenizer, text_col_name, label_col_name, class_type, num_labels, batch_size)
    val_dataset = vulDataset(val_encodings, val_labels)
    test_dataset = vulDataset(test_encodings, test_labels)
    print("DataLoader START")
    train_dataloader = DataLoader(train_dataset, batch_size=None)  # batch_size is None because it is already handled in OversampledDatasetGenerator
    val_dataloader = DataLoader(val_dataset, batch_size=None)  # batch_size is None because it is already handled in OversampledDatasetGenerator
    test_dataloader = DataLoader(test_dataset, batch_size=None)

    model, train_losses, train_accs, train_bal_accs, train_precisions, train_recalls, train_f1s = train(model, class_type, average, device, criterion, optimizer, train_dataloader, writer, step_size)
    
    # validation 
    val_loss, val_acc, val_bal_acc, val_precision, val_recall, val_f1 = evaluate(model, class_type, average, device, criterion, val_dataloader, writer)
    print(f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val Balanced Acc: {val_bal_acc:.3f}, Val F1 Score: {val_f1:.3f} Val Precision: {val_precision:.3f}, Val Recall: {val_recall:.3f}") 
    
    update_lists_and_plot(params, output_dir, train_losses, val_loss, train_accs, val_acc, train_bal_accs, val_bal_acc, train_precisions, val_precision, train_recalls, val_recall, train_f1s, val_f1)
  
    # # Save the model if it's the best one seen so far
    # if trial.best_trial is None or val_bal_acc < trial.best_trial.value:
    #     torch.save(model.state_dict(), f'{output_dir}/best_{class_type}_{model_name}_model.pth')
    
    return val_loss
        
if __name__ == "__main__":
    # fix the seed to 42
    random.seed(42)
    print(os.getcwd())
    # Load the CSV file as a Pandas DataFrame
    train_df = pd.read_csv('data_preprocessing/preprocessed_datasets/train_data.csv')
    val_df = pd.read_csv('data_preprocessing/preprocessed_datasets/val_data.csv')
    test_df = pd.read_csv('data_preprocessing/preprocessed_datasets/test_data.csv')

    frac = 0.0005
    train_df = train_df.sample(frac=frac)
    val_df = val_df.sample(frac=frac)
    test_df = test_df.sample(frac=frac)

    print(f"# of rows in train_df dataset: {train_df.shape[0]}")
    print(f"# of rows in val_df dataset: {val_df.shape[0]}")
    print(f"# of rows in test_df dataset: {test_df.shape[0]}")
    print("columns\n",train_df.columns)

    model_name = 'CodeBERT' 
    class_type = 'multi'
    n_trials = 1
    
    # Create a new directory
    output_dir = f'results/multi' if class_type == 'multi' else f'results/binary'
    output_dir = f'{output_dir}/HTO/{model_name}/n_trials{n_trials}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the study
    study = optuna.create_study(direction='minimize')

    # Run the optimization -
    study.optimize(lambda trial: objective(trial, train_df, val_df, test_df, output_dir, class_type, model_name), n_trials=n_trials)

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

    print(f"{model_name} with concatenated_df(frac={frac}) with {n_trials} trials on {class_type} classification \nHyperparameter tuning is done!!!")
    
 
