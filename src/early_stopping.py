from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, threshold=0):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_score = None

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control:TrainerControl, logs=None, **kwargs):
        eval_loss = state.log_history[-1]['eval_loss']
        score = -eval_loss
        print("INSIDE early stopping!!")
        if self.best_score is None:
            self.best_score = score
            print("self.best_score:",self.best_score)
        elif score < self.best_score + self.threshold:
            self.counter += 1
            print("self.counter:",self.counter)
            if self.counter >= self.patience:
                control.should_training_stop = True
                print("Early stopping!!!")
        else:
            self.best_score = score
            self.counter = 0


