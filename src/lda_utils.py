import re
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim_models
import pyLDAvis

def lda_analysis(df, num_topics=5):
    """
    Aplica LDA sobre la columna 'message' del dataframe
    y devuelve HTML con los tópicos.
    """
    # 1. Preprocesamiento
    stop_words = set(stopwords.words("spanish"))
    texts = [
        [w for w in word_tokenize(str(msg).lower()) if w.isalpha() and w not in stop_words]
        for msg in df['message'] if isinstance(msg, str)
    ]
    if not texts:
        return "<p>No hay suficientes mensajes para análisis LDA</p>"

    # 2. Diccionario y corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 3. Modelo LDA
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # 4. Convertir resultados a tabla HTML
    topic_html = "<h3>Temas encontrados (LDA)</h3><ul>"
    for idx, topic in lda_model.print_topics(-1):
        topic_html += f"<li><b>Tema {idx}</b>: {topic}</li>"
    topic_html += "</ul>"

    # 5. (Opcional) Visualización interactiva con pyLDAvis
    try:
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        vis_html = pyLDAvis.prepared_data_to_html(vis)
        topic_html += vis_html
    except Exception as e:
        topic_html += f"<p>Error generando pyLDAvis: {e}</p>"

    return topic_html
