import re
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo import MongoClient, errors, server_api
from pathlib import Path
from os.path import join


base_path = Path(__file__).resolve().parents[1]

def connection_bd():
    try:
        # Conexão com o MongoDB Atlas
        client = MongoClient("mongodb+srv://charlesvilela12:#ProjetoPLN7@cluster0.ieoh4ho.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        client.admin.command("ping")
        print("✅ Conexão bem-sucedida com o MongoDB Atlas!")

        db = client['TransformMind']
        return db
    except ConnectionFailure as e:
        print("❌ Erro ao conectar ao MongoDB Atlas:", e)


def find_all():
    db = connection_bd()
    collection = db['texts']
    return list(collection.find({}))

def send_text_db(df):

    db = connection_bd()
    collection = db['texts']

    dados = df.to_dict(orient='records')

    # Insere no MongoDB
    if dados:  # Garante que não está vazio
        collection.insert_many(dados)
        print(f"✅ Inseridos {len(dados)} documentos no MongoDB!")
    else:
        print("⚠️ Nenhum dado válido para inserir.")

def get_previous_questions():
    db = connection_bd()
    collection = db["semantic_cache"]
    
    # Buscar tanto as perguntas quanto as respostas
    results = collection.find({}, {"user_input": 1, "response": 1, "_id": 0})
    
    # Retornar uma lista de dicionários com 'userquestion' e 'botresponse'
    return [{"question": doc['user_input'], "response": doc['response']} for doc in results]

def insert_bd(new_interaction):
    db = connection_bd()
    collection = db["semantic_cache"]
    if collection is None:
        st.error("Não foi possível conectar ao banco de dados.")
        return
    
    data = {"user_input": new_interaction.user_question, 
             "response": new_interaction.bot_response, 
             "userid": new_interaction.user_id, 
             "timeresponse": new_interaction.timestamp,
             "datetime": datetime.now(),
             "isQuestionAudio": new_interaction.isQuestionAudio,
             "isResponseAudio": new_interaction.isResponseAudio}
    try:
        result = collection.insert_one(data)
    except errors.ServerSelectionTimeoutError as e:
        st.error(f"Erro de timeout na seleção do servidor: {e}")
    except errors.ConnectionFailure as e:
        st.error(f"Erro de conexão com o MongoDB: {e}")
    except Exception as e:
        st.error(f"Erro inesperado ao inserir dados: {e}")

def add_to_cache(user_input: str, response: str, embedding, min_similarity: float, EMBEDDING_MODEL):
    try:
        db = connection_bd()
        collection = db['semantic_cache']

        doc = {
            "user_input": user_input,
            "response": response,
            "embedding": embedding,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 0,
            "metadata": {
                "model": EMBEDDING_MODEL,
                "dimension": len(embedding),
                "similarity_threshold": min_similarity
            }
        }
    except Exception as e:
        print('⚠️ Nenhum dado válido para inserir.')


def load_bd():
    bd = find_all()
    return pd.DataFrame(bd)