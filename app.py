import faiss
import numpy as np
import streamlit as st

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from loader import load_model, load_data

st.title('Поиск похожих статей Gazeta.ru')
st.text('Система возвращает 10 результатов: краткие описания и ссылки на новости')

#загружаем модель, данные и индекс
embedder = load_model()
df = load_data()
vectors = np.stack(df['embedding'].values, axis=0)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

def get_top10(index, query):
    """Возвращает 10 статей, который максимально похожи на запрос"""
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    _, I = index.search(np.array(query_embedding).reshape((1,768)), 10)
    result = df.iloc[I[0]][['summary', 'url']].reset_index(drop=True)
    result.index = np.arange(1, len(result) + 1)
    return result

query = st.text_area("Введите запрос")
if st.button("Искать"):
    result = get_top10(index, query)
    st.write('Результат:', result)