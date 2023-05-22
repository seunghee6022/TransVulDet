from transformers import BertForSequenceClassification, BertTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification


def get_tokenizer_and_model(model_name, num_labels, dropout=0.1):
    print("num_labels for the model: ",num_labels)
    if model_name == 'BERT':
        # Load the pre-trained BERT tokenizer
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        # Load the pre-trained BERT model for sequence classification
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels, hidden_dropout_prob=dropout,attention_probs_dropout_prob=dropout)

    elif model_name == 'CodeBERT':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=num_labels,hidden_dropout_prob=dropout,attention_probs_dropout_prob=dropout)
        
    elif model_name == 'CodeRoBERTa':
        # Load the pre-trained tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/coderoberta-base")
        # Load the pre-trained model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/coderoberta-base", num_labels=num_labels,hidden_dropout_prob=dropout,attention_probs_dropout_prob=dropout)

    elif model_name == 'CodeBERTa':
        # Load the pre-trained tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codeberta-base")
        # Load the pre-trained model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/codeberta-base", num_labels=num_labels,hidden_dropout_prob=dropout,attention_probs_dropout_prob=dropout)


    elif model_name == 'T5':
        pass

    elif model_name == 'CodeT5':
        pass

    else:
        print("You put the wrong model name!!!")
    print(model)

    return tokenizer, model
