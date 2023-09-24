from transformers import Trainer
from torch.nn import BCEWithLogitsLoss

class CustomTrainer(Trainer):
    def __init__(self, use_hierarchical_classifier=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = BCEWithLogitsLoss()
        self.use_hierarchical_classifier = use_hierarchical_classifier

    # For multilabel classification, need to define the Custom Loss Function
    def compute_loss(self, model, inputs, return_outputs=False):
        print("inputs['labels']",inputs['labels'].shape)
        batch_size, num_labels = inputs['labels'].shape
        print("batch_size", batch_size, "num_labels",num_labels)

        if self.use_hierarchical_classifier:
            loss, logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
            print("logits, inputs['labels']", logits.shape,  inputs['labels'].shape)
        
        else:
            logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = self.loss_fn(logits.view(-1, num_labels), 
                            inputs['labels'].float().view(-1, num_labels))
        print("logits shape: ", logits.shape)
        print("labels shape: ", inputs['labels'].shape)
        print("loss:", loss)
        return (loss, logits) if return_outputs else loss