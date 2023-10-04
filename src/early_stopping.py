from transformers import TrainerCallback, TrainerControl, TrainingArguments

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=2, threshold=0.8):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_score = None

    def on_epoch_end(self, args: TrainingArguments, state: TrainerControl, logs=None, **kwargs):
        # Assuming we're using 'eval_loss' for metric to monitor
        eval_loss = state.log_history[-1]['eval_loss']
        score = -eval_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                state.should_training_stop = True
        else:
            self.best_score = score
            self.counter = 0


