from formatting_utils import chat_table_html
from utils import filter_system_user
from whatstk import WhatsAppChat, FigureBuilder
from whatstk.analysis import get_interventions_count
from whatstk.graph import plot
from whatstk.data import whatsapp_urls
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import regex as re  # en lugar de import re
from collections import Counter
import emoji
from wordcloud import WordCloud
import base64
from io import BytesIO
import networkx as nx
import os
import plotly.io as pio
import plotly.graph_objects as go

from transformers import pipeline

from tqdm import tqdm

from pysentimiento import create_analyzer

# Crear analizadores en español
sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
emotion_analyzer = create_analyzer(task="emotion", lang="es")


def sentiment_analysis_complete(filepath, df, min_length=3, batch_save=50):
    """
    Analiza sentimientos y emociones usando pysentimiento en español,
    guarda resultados en <filename>.sent, mostrando barra de progreso.
    """
    print("FILEPAAATH")
    output_file = f"{filepath}.sent"

    if os.path.exists(output_file):
        print(f"[INFO] Cargando análisis existente desde {output_file}")
        return pd.read_csv(output_file)

    results = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Analizando mensajes"):
        msg = str(row.message).strip()
        if len(msg.split()) < min_length or msg.startswith("<<"):
            continue

        try:
            # Sentimiento general
            sent = sentiment_analyzer.predict(msg)
            sentiment_label = sent.output  # "POS", "NEG", "NEU"
            sentiment_score = sent.probas.get(sentiment_label, 0)

            # Emociones
            emotions = emotion_analyzer.predict(msg)
            emotion_dict = {k.lower(): v for k, v in emotions.probas.items()}  # pasar a minúscula

            results.append({
                "date": row.date,
                "username": row.username,
                "message": msg,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                **emotion_dict
            })

            # Guardar resultados parciales
            if len(results) % batch_save == 0:
                pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")

        except Exception as e:
            print(f"[WARN] Error analizando mensaje: {e}")
            continue

    # Guardar resultados finales
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[INFO] Archivo de sentimientos completo generado: {output_file}")

    return res_df


def generate_polarity_statistics(df):
    html_sections = []

    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title("Distribución de Polaridad")
    plt.ylabel("Cantidad")
    plt.xlabel("Polaridad")

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    html_sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{img_base64}" /></div>')

    return html_sections


def generate_emotion_statistics(df, include_others=False):
    html_sections = []

    emotion_cols = [c for c in df.columns
                    if c not in ['date', 'username', 'message', 'sentiment', 'sentiment_score']]

    if not include_others:
        emotion_cols = [c for c in emotion_cols if c.lower() != "others"]

    emotion_means = df[emotion_cols].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=emotion_means.index, y=emotion_means.values)
    plt.title("Promedio de Emociones")
    plt.ylabel("Score promedio")
    plt.xlabel("Emoción")

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    html_sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{img_base64}" /></div>')

    return html_sections

def generate_emotion_over_time(df, include_others=False, freq="M"):
    """
    Grafica la evolución semanal de emociones y marca la semana pico de cada emoción.
    Debajo:
      - Una tabla resumen con la emoción, la semana pico y su valor.
      - Una tabla tipo chat con todos los mensajes de esas semanas top.
    - include_others: si False, excluye la columna 'others'.
    - freq: frecuencia temporal, por defecto 'W-MON' (semanas iniciando lunes).
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    html_sections = []

    # Copia segura y fechas
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Columnas de emociones
    emotion_cols = [c for c in df.columns
                    if c not in ['date', 'username', 'message', 'sentiment', 'sentiment_score']]

    if not include_others:
        emotion_cols = [c for c in emotion_cols if c.lower() != "others"]

    if not emotion_cols:
        html_sections.append("<p>No se encontraron columnas de emociones.</p>")
        return html_sections

    # Asegurar numérico
    for c in emotion_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Media semanal por emoción
    week_period = df['date'].dt.to_period(freq)
    emotion_time = df.groupby(week_period)[emotion_cols].mean()

    if emotion_time.empty:
        html_sections.append("<p>No hay datos suficientes para graficar por semanas.</p>")
        return html_sections

    # Convertir a timestamps (inicio de la semana) para plot
    emotion_time.index = emotion_time.index.to_timestamp()

    # ---- Gráfico
    plt.figure(figsize=(12, 6))
    for col in emotion_time.columns:
        plt.plot(emotion_time.index, emotion_time[col], label=col)

    # Marcar semana pico por emoción y recolectar tabla de resumen
    resumen = []
    semanas_top = set()
    for col in emotion_time.columns:
        serie = emotion_time[col].dropna()
        if serie.empty:
            continue

        # Semana pico
        top_week_ts = serie.idxmax()
        top_week_val = float(serie.loc[top_week_ts])

        # Punto grande y etiqueta en el gráfico
        plt.scatter(top_week_ts, top_week_val, s=140, marker='o',
                    edgecolor='black', facecolor='red', zorder=5)
        plt.text(top_week_ts, top_week_val, f'{col} {top_week_ts.date()}',
                 fontsize=9, ha='center', va='bottom', rotation=45)

        # Guardar resumen
        resumen.append({
            "Emoción": col,
            "Semana (inicio)": top_week_ts.date(),
            "Valor semanal medio": round(top_week_val, 3)
        })
        semanas_top.add(top_week_ts.to_period(freq))

    plt.title("Evolución de emociones (semanal) — semanas pico marcadas")
    plt.xlabel("Semana (inicio)")
    plt.ylabel("Score medio")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Guardar gráfico a base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    html_sections.append(
        f'<div class="plot-container"><img src="data:image/png;base64,{img_base64}" /></div>'
    )

    # ---- Tabla resumen
    if resumen:
        resumen_df = (pd.DataFrame(resumen)
                        .sort_values(by="Emoción")
                        .reset_index(drop=True))
        table_html = resumen_df.to_html(index=False, classes="table table-striped")
        html_sections.append("<h3>Semana pico por emoción</h3>" + table_html)

        # Agregar tabla tipo chat por cada semana pico
        df['week'] = df['date'].dt.to_period(freq)
        for row in resumen:
            emocion = row["Emoción"]
            semana_top = row["Semana (inicio)"]

            # Filtrar mensajes solo de esa semana
            mask = (df['week'] == pd.Period(semana_top, freq=freq))
            mensajes_top = df.loc[mask].copy()
            mensajes_top = mensajes_top.sort_values(by="date")

            if mensajes_top.empty:
                continue

            # Llamar a chat_table_html
            chat_html = chat_table_html(mensajes_top)
            html_sections.append(f"<h3>Mensajes en la semana pico de {emocion}</h3>" + chat_html)
    else:
        html_sections.append("<p>No se pudieron determinar semanas pico.</p>")

    return html_sections

def generate_top_emotions(df, top_n=20, include_others=False, metadata=False):
    """
    Genera secciones HTML con los top N mensajes por cada emoción.
    Cada sección usa chat_table_html para mostrar los mensajes como chat tipo WhatsApp.
    
    Args:
        df: DataFrame con columnas de fecha, usuario, mensaje y emociones.
        top_n: número de mensajes a mostrar por emoción.
        include_others: si False, excluye la columna 'others'.
        metadata: si True, muestra columnas adicionales como metadatos.
        
    Returns:
        html_sections: lista de strings HTML.
    """
    html_sections = []

    # Columnas de emociones
    emotion_cols = [c for c in df.columns
                    if c not in ['date', 'username', 'message', 'sentiment', 'sentiment_score']]

    if not include_others:
        emotion_cols = [c for c in emotion_cols if c.lower() != "others"]

    if not emotion_cols:
        html_sections.append("<p>No se encontraron columnas de emociones.</p>")
        return html_sections

    # Generar sección por emoción
    for emotion in emotion_cols:
        # Top N mensajes por emoción
        top_msgs = df.nlargest(top_n, emotion)[['date', 'username', 'message'] + ([emotion] if metadata else [])].copy()
        if metadata:
            # Redondear valor de la emoción si se muestra como metadato
            top_msgs[emotion] = top_msgs[emotion].round(3)

        # Generar tabla estilo chat
        table_html = chat_table_html(top_msgs, user_col='username', message_col='message', 
                                     date_col='date', metadata=metadata)

        html_sections.append(f"<h3>Top {top_n} mensajes con más <b>{emotion}</b></h3>{table_html}")

    return html_sections



def generate_user_emotion_pies(df, include_others=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    
    df = filter_system_user(df)

    html_sections = []

    # Columnas de emociones
    emotion_cols = [c for c in df.columns
                    if c not in ['date', 'username', 'message', 'sentiment', 'sentiment_score']]

    if not include_others:
        emotion_cols = [c for c in emotion_cols if c.lower() != "others"]

    # Agrupar por usuario y calcular promedio de emociones
    user_emotions = df.groupby('username')[emotion_cols].mean()

    for user, row in user_emotions.iterrows():
        plt.figure(figsize=(6, 6))
        plt.pie(
            row, 
            labels=row.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette("pastel", len(row))
        )
        plt.title(f"Distribución de emociones de {user}")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        html_sections.append(f'''
            <div class="plot-container" style="flex: 1 1 300px; text-align:center; margin:10px;">
                <h3>{user}</h3>
                <img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;" />
            </div>
        ''')

    # Contenedor principal con display:flex para alinearlos en fila
    html_output = f'''
    <div style="display:flex; flex-wrap: wrap; justify-content: center;">
        {"".join(html_sections)}
    </div>
    '''
    
    return html_output

def generate_top_user_per_emotion(df, include_others=False, min_messages=None):
    """
    Devuelve un string HTML con la tabla de usuarios top por emoción.
    Excluye automáticamente a usuarios con pocos mensajes.
    Si min_messages no se define, se calcula automáticamente.

    Columnas:
    - Emoción
    - Usuario top
    - Promedio
    - Porcentaje (%)
    - Nº de mensajes del usuario
    """
    import pandas as pd

    if df.empty:
        return "<p>No hay datos disponibles.</p>"

    # --- Columnas de emociones ---
    emotion_cols = [c for c in df.columns 
                    if c not in ['date', 'username', 'message', 'sentiment', 'sentiment_score']]
    if not include_others:
        emotion_cols = [c for c in emotion_cols if c.lower() != "others"]

    if not emotion_cols:
        return "<p>No se encontraron columnas de emociones.</p>"

    # --- Asegurar que las emociones sean numéricas ---
    df[emotion_cols] = df[emotion_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # --- Contar mensajes por usuario ---
    counts_per_user = df.groupby('username').size()

    # --- Calcular min_messages si no se define ---
    if min_messages is None:
        q3 = counts_per_user.quantile(0.75)
        avg_scaled = (df.shape[0] / (len(counts_per_user)) * 4)
        min_messages = min(300, int(max(q3, avg_scaled)))  # asegurar al menos 1 mensaje

    # --- Filtrar usuarios con suficientes mensajes ---
    valid_users = counts_per_user[counts_per_user >= min_messages].index
    df_filtered = df[df['username'].isin(valid_users)]

    if df_filtered.empty:
        # Mostrar top 5 usuarios con más mensajes
        top_users = counts_per_user.sort_values(ascending=False).head(5)
        top_list_html = "<ul>" + "".join([f"<li>{u}: {n} mensajes</li>" for u, n in top_users.items()]) + "</ul>"
        return f"""
        <p>No hay usuarios con al menos {min_messages} mensajes.</p>
        <p>Top 5 usuarios con más mensajes:</p>
        {top_list_html}
        """

    # --- Promedio por usuario ---
    user_emotions = df_filtered.groupby('username')[emotion_cols].mean()

    # --- Construir resultados ---
    results = []
    for emotion in emotion_cols:
        s = user_emotions[emotion]
        if s.sum() == 0:
            continue  # Ignorar emociones sin puntuación

        top_user = s.idxmax()
        top_value = s[top_user]
        n_msgs_user = counts_per_user.get(top_user, 0)

        results.append({
            'Emoción': emotion,
            'Usuario': top_user,
            'Promedio': round(top_value, 4),
            'Porcentaje (%)': round(top_value * 100, 2),
            'Mensajes usuario': n_msgs_user
        })

    # --- Si no hay resultados significativos, mostrar top 5 usuarios con más mensajes ---
    if not results:
        top_users = counts_per_user.sort_values(ascending=False).head(5)
        top_list_html = "<ul>" + "".join([f"<li>{u}: {n} mensajes</li>" for u, n in top_users.items()]) + "</ul>"
        return f"""
        <p>No se encontraron resultados significativos para las emociones.</p>
        <p>Top 5 usuarios con más mensajes:</p>
        {top_list_html}
        """

    # --- DataFrame final ---
    top_df = pd.DataFrame(results).sort_values(
        by=['Porcentaje (%)', 'Mensajes usuario'], ascending=[False, False]
    ).reset_index(drop=True)

    # --- HTML ---
    html_table = top_df.to_html(index=False, classes="table table-striped", border=0, escape=False)
    html_string = f"""
    <div class="table-container" style="max-height:400px; overflow-y:auto;">
        <p style="font-style:italic; color:#555;">Mensajes mínimos requeridos por usuario: {min_messages}</p>
        <h3>Usuarios top por emoción</h3>
        {html_table}
    </div>
    """
    return html_string

