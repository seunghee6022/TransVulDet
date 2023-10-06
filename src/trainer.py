from transformers import Trainer
from torch.nn import BCEWithLogitsLoss

class CustomTrainer(Trainer):
    '''
    #class_weights --> loss gets weighted by the ground truth class -- inverse probabiity weighting
    e.g. def __init__(self, use_hierarchical_classifier=False, class_weights, *args, **kwargs): 

    Or find other weight loss and combine with comepute_loss!

    Or BCEWithLogitsLoss + pos_weight arg  --> Binary classification
    
    '''
    def __init__(self, use_hierarchical_classifier, uid_to_dimension, *args, **kwargs): 
        super().__init__( *args, **kwargs)
        self.loss_fn = BCEWithLogitsLoss()
        self.use_hierarchical_classifier = use_hierarchical_classifier
        self.uid_to_dimension = uid_to_dimension
        self.num_labels = len(uid_to_dimension)
        # print("CustomTrainer is Initialized!!!!!!!!!!!")
        # self.print_model()

    def one_hot_encode(self, labels):
        print("labels:",labels)
        one_hot_encoded = []
        for label in labels:
            one_hot = [0] * self.num_labels
            one_hot[self.uid_to_dimension[label]] = 1
            one_hot_encoded.append(one_hot) 
        print("one_hot_encoded",one_hot_encoded) 
        return one_hot_encoded
    
    # For multilabel classification, need to define the Custom Loss Function
    def compute_loss(self, model, inputs, return_outputs=False):
        print("THIS IS COMPUTE LOSS IN TRAINER")
        print("inputs",type(inputs), inputs)
        print("inputs['labels']",inputs['labels'].shape)
        batch_size, num_labels = len(inputs['labels']), self.num_labels
        print("batch_size, num_labels:",batch_size, num_labels)

        if self.use_hierarchical_classifier:
            # loss, logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
            logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            print("logits.view(-1, num_labels)", logits.shape, logits.view(-1, num_labels).shape,logits.view(-1, num_labels) )
            loss = model.loss(logits, inputs['labels'])
            # print("logits, inputs['labels']", logits.shape,  inputs['labels'].shape)
        
        else:
            logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = self.loss_fn(logits.view(-1, num_labels), inputs['labels'].float().view(-1, num_labels))
            
        # print("logits shape: ", logits.shape)
        # print("labels shape: ", inputs['labels'].shape)
        # print("loss:", loss)
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