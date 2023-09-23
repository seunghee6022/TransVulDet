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
        
        model_num_labels = model.model.config.num_labels if self.use_hierarchical_classifier else model.config.num_labels
        print(model_num_labels)
        # Only reshape if the number of labels doesn't match the model's config
        if num_labels != model_num_labels:
            print(f"num_labels {num_labels}!= model_num_labels {model_num_labels}")

        logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        print("logits shape: ", logits.shape)
        print("labels shape: ", inputs['labels'].shape)
        loss = self.loss_fn(logits.view(-1, num_labels), 
                        inputs['labels'].float().view(-1, num_labels))
        
        return (loss, logits) if return_outputs else loss