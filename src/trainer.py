from transformers import Trainer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from focal_loss.focal_loss import FocalLoss
import torch
import torch.nn.functional as F
import numpy as np


class CustomTrainer(Trainer):
    '''
    #class_weights --> loss gets weighted by the ground truth class -- inverse probabiity weighting
    e.g. def __init__(self, use_hierarchical_classifier=False, class_weights, *args, **kwargs): 

    Or find other weight loss and combine with comepute_loss!

    Or BCEWithLogitsLoss + pos_weight arg  --> Binary classification
    
    '''
    def __init__(self, use_hierarchical_classifier, prediction_target_uids, use_focal_loss, use_bilstm, class_weights, *args, **kwargs): 
        super().__init__( *args, **kwargs)
        self.loss_fn = CrossEntropyLoss()
        self.use_hierarchical_classifier = use_hierarchical_classifier
        self.prediction_target_uids = prediction_target_uids
        self.use_focal_loss = use_focal_loss
        self.target_to_dimension = {target:idx for idx,target in enumerate(self.prediction_target_uids)}
        self.class_weights = class_weights
        self.use_bilstm = use_bilstm
        
    def mapping_cwe_to_label(self, cwe_label):
        # Convert each tensor element to its corresponding dictionary value
        mapped_labels = [self.target_to_dimension[int(cwe_id.item())] for cwe_id in cwe_label]
        return torch.tensor(mapped_labels)
    
    # For multilabel classification, need to define the Custom Loss Function
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.use_hierarchical_classifier:
            input_without_labels = {k: v for k, v in inputs.items() if k != 'labels'}
            logits = model(**input_without_labels)
            loss = model.loss(logits, inputs['labels'])
            
        
        else:
            input_without_labels = {k: v for k, v in inputs.items() if k != 'labels'}
            logits = model(**input_without_labels)
            if not self.use_bilstm :
                logits = logits.logits #[batch_size, sequence_length, num_classes] logits, inputs['labels'] torch.Size([6, 21]) torch.Size([6])
            
            # Apply softmax to logits to get probabilities
            logits = F.softmax(logits, dim=-1).cpu()
            labels = self.mapping_cwe_to_label(inputs['labels'].cpu())

            if self.use_focal_loss:
                gamma = 2.0
                self.loss_fn = FocalLoss(gamma=gamma, weights=self.class_weights.cpu()) #https://pypi.org/project/focal-loss-torch/
                loss = self.loss_fn(logits, labels)
            else:
                loss = self.loss_fn(logits, labels)

        return (loss, (loss,logits)) if return_outputs else loss
