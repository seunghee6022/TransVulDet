import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from .src.dataset import *
import matplotlib.pyplot as plt

# Load the CSV file as a Pandas DataFrame
df = pd.read_csv('/home/jeon_su/Desktop/TransVulDet/data preprocessing/preprocessed datasets/MSR.csv')
print(df.columns)

# Split the dataset into training, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(get_texts(df['func_before']), get_labels(df['vul']), test_size=0.2)
train_texts, test_texts, train_labels, test_labels = train_test_split(get_texts(df['func_before']), get_labels(df['vul']), test_size=0.2)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

# Load the pre-trained model and tokenizer
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(df['vul'].tolist())))

# Tokenize the input texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


train_dataset = vulDataset(train_encodings, train_labels)
val_dataset = vulDataset(val_encodings, val_labels)
test_dataset = vulDataset(test_encodings, test_labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)

# Define the trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

train_result = trainer.train()
eval_result = trainer.evaluate(test_dataset)

# Print the evaluation results
print("***** Test results *****")
for key, value in eval_result.items():
    print(f"{key} = {value}")

# Compute predictions and evaluation metrics on the test set
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(-1)
accuracy = accuracy_score(test_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Plot the training and validation loss and accuracy curves
train_loss = train_result.loss
train_acc = train_result.metrics['train_accuracy']
eval_loss = eval_result['eval_loss']
eval_acc = eval_result['eval_accuracy']

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, eval_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'g', label='Training accuracy')
plt.plot(epochs, eval_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
