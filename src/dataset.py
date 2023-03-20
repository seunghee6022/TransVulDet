import torch
from torch.utils.data import Dataset

def get_labels(df):
    unique_values = set(df.tolist())

    # Create a dictionary to map the unique values to integer indices
    value_to_index = {value: index for index, value in enumerate(unique_values)}

    # Create a list of one-hot encoded labels for each row of the DataFrame
    labels = []
    for value in df:
        index = value_to_index[value]
        one_hot_label = torch.zeros(len(unique_values))
        one_hot_label[index] = 1
        labels.append(one_hot_label)
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


