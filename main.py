from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from torch import nn

import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import gensim.downloader as api
import numpy as np


class SVMModel:
    def __init__(self, kernel='linear', C=1.0):
        self.model = SVC(kernel=kernel, C=C)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class NaiveBayesModel:
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class MLPModel:
    def __init__(self):
        self.model = MLPClassifier()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
class XGBoostModel:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze(0)  
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_model(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        step = 0
        bar = tqdm(train_loader)
        for batch in bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            step += 1
            total_loss += loss.item()
            accuracy = (torch.argmax(outputs.logits, dim=1) == labels).float().mean()
            # bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f, step: {step}}")
            bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Step: {step}, Accuracy: {accuracy:.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def get_word2vec_embeddings(texts, word2vec_model, embedding_dim=300):
    embeddings = []
    for text in texts:
        word_vectors = [word2vec_model[word] for word in text if word in word2vec_model]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(embedding_dim))
    return np.array(embeddings)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds))


def save_model(model, path):
    model.save_pretrained(path)
    print(f"Model saved to {path}")

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='naive_bayes', help='Model to run')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='C:/Users/16366/ECE449_Project/email_text.csv', help='Path to data')
    parser.add_argument('--save_path', type=str, default='C:/Users/16366/ECE449_Project/bert_model/', help='Path to save model')
    parser.add_argument('--pretrained_model', type=str, default='/bert-base-uncased', help='Path to pretrained model')


    args = parser.parse_args()

    model = args.model
    if args.model not in ['svm', 'logistic', 'bert', 'naive_bayes', 'mlp',"xgboost"]:
        print("Invalid model")
        return
    
    df = pd.read_csv(args.data_path)

    # texts = [
    #     "I love programming", "Python is amazing", "I hate bugs",
    #     "Debugging is frustrating", "I enjoy machine learning", "AI is the future"
    # ]
    # labels = [1, 1, 0, 0, 1, 1]

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # texts = texts[:5000]
    # labels = labels[:5000]
    print("Number of texts:", len(texts))
    print("Number of labels:", len(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42
    )

    # svm
    if model == 'svm':
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        svm_model = SVMModel(kernel='linear')
        svm_model.fit(X_train_tfidf, y_train)
        svm_preds = svm_model.predict(X_test_tfidf)
        print("SVM Results:", classification_report(y_test, svm_preds))
   
    # logistic regression
    elif model == 'logistic':
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logistic_model = LogisticRegressionModel()
        logistic_model.fit(X_train_tfidf, y_train)
        logistic_preds = logistic_model.predict(X_test_tfidf)
        print("Logistic Regression Results:", classification_report(y_test, logistic_preds))

    # Naive Bayes
    elif model == 'naive_bayes':
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        nb_model = NaiveBayesModel()
        nb_model.fit(X_train_tfidf, y_train)
        nb_preds = nb_model.predict(X_test_tfidf)
        print("Naive Bayes Results:", classification_report(y_test, nb_preds))

    # MLP
    elif model == 'mlp':
        word2vec_model = api.load("word2vec-google-news-300")
        X_train_dense = get_word2vec_embeddings(X_train, word2vec_model)
        X_test_dense = get_word2vec_embeddings(X_test, word2vec_model)
        mlp_model = MLPModel()
        mlp_model.fit(X_train_dense, y_train)
        mlp_preds = mlp_model.predict(X_test_dense)
        print("MLP Results:", classification_report(y_test, mlp_preds))
        
    elif model == 'xgboost':
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        xgb_model = XGBoostModel()
        xgb_model.fit(X_train_tfidf, y_train)
        xgb_preds = xgb_model.predict(X_test_tfidf)
        print("XGBoost Results:", classification_report(y_test, xgb_preds))




    else:
        model = BertForSequenceClassification.from_pretrained(args.pretrained_model,num_labels=2)
        model.to(device)

        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
        train_dataset = TextDataset(X_train, y_train, tokenizer)
        test_dataset = TextDataset(X_test, y_test, tokenizer)

        train_loader_bert = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
        test_loader_bert = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        print("BERT Results:")
        train_model(model, train_loader_bert, optimizer, criterion, epochs=args.epochs, device=device)
        evaluate_model(model, test_loader_bert, device=device)
        save_model(model, args.save_path)


if __name__ == "__main__":
    main()