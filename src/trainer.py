from transformers import Trainer
from torch.nn import BCEWithLogitsLoss

class CustomTrainer(Trainer):
    def __init__(self, use_hierarchical_classifier=False, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.loss_fn = BCEWithLogitsLoss()
        # self.model = model
        self.use_hierarchical_classifier = use_hierarchical_classifier
        print("CustomTrainer is Initialized!!!!!!!!!!!")
        self.print_model()

    # For multilabel classification, need to define the Custom Loss Function
    def compute_loss(self, model, inputs, return_outputs=False):
        print("THIS IS COMPUTE LOSS IN TRAINER")
        # print("inputs['labels']",inputs['labels'].shape)
        batch_size, num_labels = inputs['labels'].shape
        print("batch_size", batch_size, "num_labels",num_labels)

        if self.use_hierarchical_classifier:
            print("self.use_hierarchical_classifier:",{self.use_hierarchical_classifier})
            # loss, logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
            logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = model.loss(logits, inputs['labels'])
            print("logits, inputs['labels']", logits.shape,  inputs['labels'].shape)
        
        else:
            logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = self.loss_fn(logits.view(-1, num_labels), inputs['labels'].float().view(-1, num_labels))
        print("logits shape: ", logits.shape)
        print("labels shape: ", inputs['labels'].shape)
        print("loss:", loss)
        return (loss, (loss,logits)) if return_outputs else loss

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        print("INSIDE log_metrics---metrics:",metrics)
        # Add accuracy and F1 score to the logs
        if "accuracy" in metrics:
            self.state.log_history.append({"eval_accuracy": metrics["accuracy"], "step": step})
        if "f1_score" in metrics:
            self.state.log_history.append({"eval_f1_score": metrics["f1_score"], "step": step})

    def print_model(self):
        print(self.model)