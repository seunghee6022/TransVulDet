import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset, IterableDataset
from sklearn.model_selection import train_test_split

def split_dataframe(df_path, test_size=0.3,random_state=42):
    df = pd.read_csv(df_path)
    # Split data into train and temp (which will be further split into val and test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

    # Split temp_df into validation and test datasets
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df

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


def get_texts(df):
    # Preprocess the data as necessary (e.g., get the texts from the DataFrame)
    texts = df.tolist()
    return texts

# Create a PyTorch dataset
class vulDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels
    print("Inside vulDataset!!!!!!!!!!!!!")
    print("self.encodings", self.encodings)
    print("self.labels", self.labels)

  def __getitem__(self, idx):

    item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    #item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32).clone().detach()
    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32).clone().detach().requires_grad_(True)

    return item

  def __len__(self):
    return len(self.labels)
  

class OversampledDatasetGenerator(IterableDataset):
    def __init__(self, df, tokenizer, text_col_name, label_col_name, class_type, num_labels, batch_size=8):
        self.df = shuffle(df)
        self.tokenizer = tokenizer
        self.text_col_name = text_col_name
        self.label_col_name = label_col_name
        self.class_type = class_type
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.oversampler = RandomOverSampler(sampling_strategy="minority", random_state=42) if class_type=="binary" else RandomOverSampler(random_state=42)
        
    def __iter__(self):
        for i in range(0, self.df.shape[0], self.batch_size):
            batch = self.df.iloc[i:i+self.batch_size]

            texts = get_texts(batch[self.text_col_name])
            labels = get_labels(batch[self.label_col_name], self.num_labels)

            if self.class_type == 'multi':
                labels = one_hot_to_labels(labels)
            min_samples = min(len(texts), len(labels))
            texts = texts[:min_samples]
            labels = labels[:min_samples]
            texts = np.array(texts).reshape(-1, 1)
            print("Before labels in OversampledDatasetGenerator:",labels)
            unique_classes = np.unique(labels)
            print(f"unique_classes:{unique_classes}")

            # binary_continue_cond = self.class_type == 'binary' and len(unique_classes) <= 1
            # multi_continue_cond = self.class_type == 'multi' and len(unique_classes) <= 3

            if self.class_type == 'binary':
                if len(unique_classes) <= 1:
                    continue
                else:
                    if labels.count(1) <=2:
                        continue
            else:
                if len(unique_classes) <= 3:
                    continue

            counter = Counter(labels)
            print(counter)

            # Oversample only if there's more than one class.
            resampled_texts, resampled_labels = self.oversampler.fit_resample(texts, labels)
            print("After labels in OversampledDatasetGenerator:",resampled_labels)
                
            resampled_encodings = self.tokenizer(list(resampled_texts.flatten()), truncation=True, padding=True, return_tensors='pt')

            if self.class_type == 'multi':
                resampled_one_hot_labels = torch.eye(self.num_labels)[resampled_labels]
                resampled_labels = resampled_one_hot_labels

            yield resampled_encodings, resampled_labels
            
       
class CodeDataset(Dataset):
    def __init__(self, encodings, labels, uid_to_dimension):
        self.encodings = encodings
        self.labels = labels
        self.uid_to_dimension = uid_to_dimension
        self.num_classes = len(uid_to_dimension)
        self.one_hot_labels = self.one_hot_encode(labels)

    def one_hot_encode(self, labels):
        one_hot_encoded = []
        for label in labels:
            one_hot = [0] * self.num_classes
            if label in self.uid_to_dimension:
                one_hot[self.uid_to_dimension[label]] = 1
            else:
                print(f"Warning: Label {type(label)}{label} not found in uid_to_dimension!")
                if ', CWE' in label:
                    print("Wrong label", label)
                    continue
                else:
                    print("String label:",label)
                    label = int(label)
            
            one_hot_encoded.append(one_hot)
        return one_hot_encoded

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.one_hot_labels[idx])
        return item

    def __len__(self):
        return len(self.labels)