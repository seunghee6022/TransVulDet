import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def get_labels(df, num_labels):
    # load dict to map the unique values to integer indices
 
    # Create a list of one-hot encoded labels for each row of the DataFrame
    labels = []
    
    # one-hot-encoded
    for index in df:
        one_hot_label = torch.zeros(num_labels)
        one_hot_label[index] = 1
        labels.append(one_hot_label)

    return labels


def one_hot_to_labels(one_hot_tensor):
    # Convert to label tensor
    labels = torch.argmax(one_hot_tensor, dim=1)

    # Convert to numpy array
    labels = labels.numpy()

    return labels

def get_CVEfixes_labels(df, num_labels):
   
    # load dict to map the unique values to integer indices
    with open("data_preprocessing/preprocessed_datasets/total_cwe_dict.txt", "rb") as myFile:
        total_cwe_dict = pickle.load(myFile)

    # replace CWE IDs based on dictionary, drop the rows if CWE ID is not a key
    # multi-class
    if num_labels > 2:
       
        print("label df type is ",type(df))
      
        # NaN : 0, else map to the total_cwe_dict
        labels = df.apply(lambda x: total_cwe_dict.get(x) if x in total_cwe_dict.keys() else 0).astype(int).tolist()
        print("multi",df.head(5), labels[:5])
        one_hot_labels = torch.eye(num_labels)[labels]
        return one_hot_labels
    # binary class    
    else:
        labels = df.apply(lambda x: 1 if x in total_cwe_dict else 0).astype(int).tolist()
        print("binary",df.head(5), labels[:5])
        return labels


def get_texts(df):
    # Preprocess the data as necessary (e.g., get the texts from the DataFrame)
    texts = df.tolist()
    return texts


# Create a PyTorch dataset
class vulDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels


  def __getitem__(self, idx):
  
    item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32).clone().detach().requires_grad_(True)

    return item

  def __len__(self):
    return len(self.labels)
