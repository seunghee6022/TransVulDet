from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pickle
import pandas as pd
import sqlite3 as lite
from sqlite3 import Error
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.dataset import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("-------------------------Start DB connection--------------------")
def create_connection(db_file):
    """
    create a connection to sqlite3 database
    """
    conn = None
    try:
        conn = lite.connect(db_file, timeout=10)  # connection via sqlite3
        # engine = sa.create_engine('sqlite:///' + db_file)  # connection via sqlalchemy
        # conn = engine.connect()
    except Error as e:
        print(e)
    return conn


DATA_PATH = Path.cwd().parents[0] / 'CVEfixes'
FIGURE_PATH = Path.cwd() / 'figures'
RESULT_PATH = Path.cwd() / 'results'

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(FIGURE_PATH).mkdir(parents=True, exist_ok=True)
Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)

conn = create_connection(DATA_PATH / "CVEfixes.db")

query = """
SELECT f.code_before, f.code_after, cc.cwe_id, cw.cwe_name, mc.code
FROM file_change f, commits c, fixes fx, cve cv, cwe_classification cc, method_change mc, cwe cw
WHERE f.hash = c.hash 
AND c.hash = fx.hash 
AND fx.cve_id = cv.cve_id 
AND cv.cve_id = cc.cve_id 
AND cc.cwe_id = cw.cwe_id
AND f.file_change_id = mc.file_change_id

"""
print("-------------------------Query the data and get the dataframe--------------------")
# Execute the SQL query and fetch the results
CVEfixes_df = pd.read_sql_query(query, conn)

print("-------------------------split the dataset from df--------------------")
codes, labels = get_texts(CVEfixes_df['code']), get_labels(CVEfixes_df['cve_id'])
train_codes, val_codes, train_labels, val_labels = train_test_split(codes, labels, test_size=.2)
# train_codes, val_codes, train_labels, val_labels = train_test_split(train_codes, train_labels, test_size=.2)

class vulDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



# Initialize the tokenizer for the CodeBERT model
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

train_encodings = tokenizer(train_codes, truncation=True, padding=True)
val_encodings = tokenizer(val_codes, truncation=True, padding=True)
# test_encodings = tokenizer(test_codes, truncation=True, padding=True)

# Initialize the custom dataset with the retrieved data and tokenizer
train_dataset = vulDataset(train_encodings, train_labels)
val_dataset = vulDataset(val_encodings, val_labels)
# test_dataset = vulDataset(test_encodings, test_labels)

# Define the data loader with a batch size of 8
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


# get cwe label dictionary
with open("data preprocessing/preprocessed datasets/total_cwe_dict.txt", "rb") as myFile:
    total_cwe_dict = pickle.load(myFile)

labels = list(total_cwe_dict.keys())
print("Total # of cwe ids: ",len(labels))

num_labels = len(labels)

# Load the CodeBERT model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = AutoModelForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=num_labels)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()


# Define the evaluation function
def evaluate(model, eval_dataset):
    model.eval()
    eval_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == labels).sum().item()
    avg_loss = total_loss / len(eval_dataset)
    accuracy = total_correct / len(eval_dataset)
    return avg_loss, accuracy

# Evaluate the model and print the results
avg_loss, accuracy = evaluate(model, val_dataset)
print("Evaluation Results:")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Accuracy: {accuracy:.2%}")

# Plot the loss curve
plt.plot(avg_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(f"output/CVEfixes_training_result_avgloss_{avg_loss:.4f}_acc_{accuracy:.2%}", format='png')
plt.show()
