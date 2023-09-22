from transformers import Trainer
from torch.nn import BCEWithLogitsLoss

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = BCEWithLogitsLoss()  

    # For multilabel classification, need to define the Custom Loss Function
    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs['labels'].shape)
        batch_size, num_labels = inputs['labels'].shape
        print(batch_size, num_labels)
        # Only reshape if the number of labels doesn't match the model's config
        if num_labels != model.config.num_labels:
            print("num_labels != model.config.num_labels")

        logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        
        loss = self.loss_fn(logits.view(-1, model.config.num_labels), 
                        inputs['labels'].float().view(-1, model.config.num_labels))
        return (loss, logits) if return_outputs else loss