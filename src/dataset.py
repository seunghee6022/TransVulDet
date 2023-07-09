import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset, IterableDataset


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
         
            unique_classes = np.unique(labels)
            if len(unique_classes) > 1:
                # Oversample only if there's more than one class.
                resampled_texts, resampled_labels = self.oversampler.fit_resample(texts, labels)
            else:
                # Skip oversampling.
                resampled_texts, resampled_labels = texts, labels
            resampled_encodings = self.tokenizer(list(resampled_texts.flatten()), truncation=True, padding=True, return_tensors='pt')

            if self.class_type == 'multi':
                resampled_one_hot_labels = torch.eye(self.num_labels)[resampled_labels]
                resampled_labels = resampled_one_hot_labels

            yield resampled_encodings, resampled_labels
