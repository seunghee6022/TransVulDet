import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")