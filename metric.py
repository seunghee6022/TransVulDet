import os
import json
import matplotlib.pyplot as plt


print(os.getcwd())

# Define the path to the output directory
output_dir = 'output/checkpoint-290/'

# Load the training and validation loss from the output directory
with open(os.path.join(output_dir, 'trainer_state.json'), 'r') as f:
    train_results = json.load(f)

print(train_results.keys())
log_history = train_results['log_history']
print(log_history[0])

best_metric = train_results['best_metric']
print(best_metric)

train_loss, eval_loss = [], []
lrs = []
lr = '{:.0e}'.format(log_history[0]['learning_rate'])
for log in log_history:
    if 'loss' in log:
        train_loss.append(log['loss'])
    if 'eval_loss' in log:
        eval_loss.append(log['eval_loss'])
    if 'learning_rate' in log:
        lr = log['learning_rate']
        short_lr = '{:.0e}'.format(lr)
        lrs.append(short_lr)

# Training and Validation Loss with best metric
plt.figure(0)
plt.plot(train_loss, label='Train Loss')
plt.plot(eval_loss, label='Eval Loss')
plt.axhline(y=best_metric, color='r', linestyle='--', label='Best metric')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss')
plt.legend()
plt.show()

# Learning rate changes by weight decay
plt.figure(1)
plt.plot(lrs, label='Learning rate')
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.title(f'Learning rate (weight decay=0.01)')
plt.legend()
plt.show()





