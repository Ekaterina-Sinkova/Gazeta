import pandas as pd
from sentence_transformers import SentenceTransformer


embedder = None
df = None

def load_model():
    """Load the model"""
    global embedder
    if not embedder:
        embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return embedder

def load_data():
    """Load the data"""
    global df
    if not df:
        df1 = pd.read_pickle('my_dataframe1.pkl')
        df2 = pd.read_pickle('my_dataframe2.pkl')
        df3 = pd.read_pickle('my_dataframe3.pkl')
        return pd.concat([df1, df2, df3], axis=0)