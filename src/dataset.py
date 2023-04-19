import pickle
import torch
from torch.utils.data import Dataset

'''
def get_labels(df):
    # load dict to map the unique values to integer indices
    with open("data preprocessing/preprocessed datasets/total_cwe_dict.txt", "rb") as myFile:
        total_cwe_dict = pickle.load(myFile)

    # Define the classification labels
    labels_list = list(total_cwe_dict.keys())

    # Create a list of one-hot encoded labels for each row of the DataFrame
    labels = []
    for value in df:
        index = total_cwe_dict[value]
        labels.append(index)
    return labels
'''


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
  
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx]).to(torch.float32)
    
    return item

  def __len__(self):
    return len(self.labels)
