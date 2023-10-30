from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import wandb
import numpy as np
import optuna

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
            print("self.best_score:",-self.best_score)
        elif score < self.best_score + self.threshold:
            self.counter += 1
            print("self.counter:",self.counter)
            if self.counter >= self.patience:
                control.should_training_stop = True
                print("Early stopping!!!")
        else:
            self.best_score = score
            self.counter = 0


class WandbCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control:TrainerControl, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                wandb.log({"train_loss": logs["loss"], "learning_rate":logs["learning_rate"], "epoch":logs["epoch"]})

class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial, args):
        self.trial = trial
        self.max_steps = args.max_evals
        self.metric_for_best_model = f"eval_{args.eval_metric}"

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control:TrainerControl, logs=None, **kwargs):
        step = state.global_step

        # Stop the training if step > max_steps
        if step > self.max_steps:
            print("control.should_training_stop",control.should_training_stop)
            control.should_training_stop = True
            return
        
        metrics = state.log_history[-1].get(self.metric_for_best_model, None)
        print(f"metric log_history:{metrics}")
        
        # Report intermediate value to optuna
        if metrics is not None:
            self.trial.report(metrics, step)

            # Prune trial if need be
            if self.trial.should_prune():
                message = "Trial was pruned at iteration {}.".format(step)
                raise optuna.TrialPruned(message)
        


