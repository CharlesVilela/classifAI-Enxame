from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mealpy.swarm_based.ABC import OriginalABC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pyswarms as ps
import numpy as np
import pandas as pd
import dao.connection_bd as connection_bd
import algorithm.algorithm_pso as algorithm_pso
import algorithm.algorithm_abc as algorithm_abc
import algorithm.algorithm_gwo as algorithm_gwo
import algorithm.algorithm_fa as algorithm_fa


# Download recursos (executar uma vez)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_with_nltk(texts):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_texts = []
    
    for text in texts:
        tokens = word_tokenize(text.lower())
        tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in stop_words
        ]
        processed_texts.append(tokens)
        
    return processed_texts

def map_score_to_label(score):
    if score <= 0.33:
        return 'low'
    elif score <= 0.66:
        return 'average'
    else:
        return 'high'


def main():

    raw_data = connection_bd.find_all()
    data = pd.DataFrame(raw_data)

    # # 1. Pré-processamento
    print("🛠️ Pré-processando documentos...")
    processed_tokens = preprocess_with_nltk(data["text"])
    data["text_clean"] = [' '.join(tokens) for tokens in processed_tokens]

    # data['maturity_label'] = data['maturity_score'].apply(map_score_to_label)
    # build_pipeline_complete(data['text_clean'],data['maturity_label'], "model_train_maturity_score")

    intent_counts = data['intent'].value_counts()
    valid_intents = intent_counts[intent_counts >= 2].index
    df_filtered = data[data['intent'].isin(valid_intents)]

    line_length = 70
    # print("# ","="*line_length)
    # print("# 🔹 Função build_pipeline_complete atualizada com PSO")
    # print("# ","="*line_length)
    # algorithm_pso.build_pipeline_complete_pso(df_filtered['text_clean'], df_filtered['intent'])
    
    # print("# ","="*line_length)
    # print("# 🔹 Função build_pipeline_complete atualizada com ABC")
    # print("# ","="*line_length)
    # algorithm_abc.build_pipeline_complete_abc(df_filtered['text_clean'], df_filtered['intent'])

    # print("# ","="*line_length)
    # print("# 🔹 Função build_pipeline_complete atualizada com GWO")
    # print("# ","="*line_length)
    # algorithm_gwo.build_pipeline_complete_gwo(df_filtered['text_clean'], df_filtered['intent'])

    print("# ","="*line_length)
    print("# 🔹 Função build_pipeline_complete atualizada com FA")
    print("# ","="*line_length)
    algorithm_fa.build_pipeline_complete_fa(df_filtered['text_clean'], df_filtered['intent'])
   

if __name__ == "__main__":
    main()
