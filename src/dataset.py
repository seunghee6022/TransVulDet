import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle

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

    #for value in df: #one-hot-encoded label
        # index = total_cwe_dict[value]
        # one_hot_label = torch.zeros(len(labels_list))
        # one_hot_label[index] = 1
        # labels.append(one_hot_label)
    return labels


def get_texts(df, model_name=None):
    # Preprocess the data as necessary (e.g., get the texts from the DataFrame)
    texts = df.tolist()
    if model_name == "CodeT5" or model_name == "T5":
        # Define the classification task
        task = 'classification'
        prefix = task + ':'
        texts = [prefix + text for text in texts]
        return texts
    
    elif model_name == "GPT2" or model_name == "CodeGPT2":
        pass

    return texts

# Create a PyTorch dataset
class vulDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)





class MyDataset1(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data.iloc[idx, :-1].values.astype('float32')
        label = self.data.iloc[idx, -1:].values.astype('int64')
        return code, label


csv_file_1 = 'path/to/csv1'
dataset1 = MyDataset1(csv_file_1)
dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=True)



