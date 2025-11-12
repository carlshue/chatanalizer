# ------------------------
# Manipulación de datos
# ------------------------
import pandas as pd  # manejo de DataFrames
from collections import Counter  # contar elementos fácilmente
import os  # manejo de rutas y archivos
from config import *
import pandas as pd
import copy
# ------------------------
# Expresiones regulares y emojis
# ------------------------
import regex as re  # expresiones regulares avanzadas (soporta \X para emojis)
import emoji  # detección y manejo de emojis

# ------------------------
# Visualización
# ------------------------
import matplotlib.pyplot as plt  # gráficos tradicionales
import seaborn as sns  # visualización estadística
import plotly.express as px  # gráficos interactivos simples
import plotly.graph_objects as go  # gráficos interactivos más complejos
import plotly.io as pio  # configuración y renderizado de plotly

# ------------------------
# Procesamiento de texto y NLP
# ------------------------
from wordcloud import WordCloud  # generar nubes de palabras
from transformers import pipeline  # modelos de NLP, p.ej. análisis de sentimiento
from tqdm import tqdm  # barras de progreso

# ------------------------
# Análisis de chats WhatsApp
# ------------------------
from whatstk import WhatsAppChat, FigureBuilder
from whatstk.analysis import get_interventions_count
from whatstk.graph import plot
from whatstk.data import whatsapp_urls
from formatting_utils import chat_table_html  # funciones de formato de tablas HTML
from scipy.optimize import curve_fit
# ------------------------
# Otros útiles
# ------------------------
from io import BytesIO  # trabajar con buffers de memoria (para imágenes)
import base64  # codificación/decodificación de imágenes en base64
import networkx as nx  # análisis de redes (gráficos de nodos)

VERBOSE = get_config().get("VERBOSE", False)


# -------------------------------
# Función para cargar chat
def load_chat(filepath):
    #ensure the file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe.")
    
    #ensure file is not empty
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"El archivo {filepath} está vacío.")
    
    #if verbose, print some info
    if VERBOSE:
        #prompt file lenght
        file_size = os.path.getsize(filepath)
        print(f"El archivo {filepath} tiene un tamaño de {file_size} bytes.")
        
        #prompt first 5 lines
        print("First lines:")
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in range(5):
                print(f.readline().strip())
    
    
    print('Loading...' + filepath)
    
    
    chat = WhatsAppChat.from_source(filepath=filepath)    #ERROR
    print(chat.df['date'])
    chat.df['date'] = pd.to_datetime(chat.df['date'])
    return chat, chat.df.copy()




#sampleeeeee intentando arreglarlo
'''
    print('Loading...' + filepath)
    
    try: #TODO: TE JAS QIEDADO POR AQUI INTENTANDO ARREGLAR LAS FECHAS DE MAGISTERIO, AUNQUE CON DIFUSION FIESTA TE FUNCIONA BIEN
        print("\n[DEBUG] Cargando con formato forzado mm/dd/yy...")
        chat = WhatsAppChat.from_source(filepath=filepath, hformat="mm/dd/yy, h:mm AM/PM")
        print("[DEBUG] Carga exitosa ✅")
        chat.df["date"] = pd.to_datetime(chat.df["date"])
        return chat, chat.df.copy()
    
    except Exception as e:
            print("\n❌ Error al parsear con formato corto (mm/dd/yy):")
            traceback.print_exc()

            print("\n[DEBUG] Reintentando con formato largo (mm/dd/yyyy)...")
            try:
                chat = WhatsAppChat.from_source(filepath=filepath, hformat="mm/dd/yyyy, h:mm AM/PM")
                print("[DEBUG] Carga exitosa con yyyy ✅")
                chat.df["date"] = pd.to_datetime(chat.df["date"])
                return chat, chat.df.copy()
            except Exception as e2:
                print("\n❌ También falló con yyyy:")
                traceback.print_exc()

                # Intento extra: detectar manualmente la línea problemática
                print("\n[DEBUG] Buscando línea con fecha inválida...")
                import re
                date_re = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4}),")
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, start=1):
                        m = date_re.match(line)
                        if m:
                            month, day = int(m.group(1)), int(m.group(2))
                            if not (1 <= month <= 12 and 1 <= day <= 31):
                                print(f"⚠️  Línea {i} sospechosa: {repr(line)}")

                raise e2  # vuelve a lanzar el error si nada funcionóç
'''









def remove_system_from_wa_chat(chat):
    """
    Devuelve una copia del chat sin los mensajes del sistema (<<system>>)
    """
    chat_copy = copy.deepcopy(chat)

    df = chat_copy.df.copy()
    if "username" not in df.columns:
        raise KeyError(f"El DataFrame no tiene columna 'username'. Columnas disponibles: {df.columns.tolist()}")

    # Filtramos los mensajes que no son del sistema
    df_filtered = df[df["username"] != "<<system>>"].reset_index(drop=True)

    # Si el objeto WhatsAppChat se construye con from_dataframe, la usamos:
    if hasattr(chat_copy.__class__, "from_dataframe"):
        chat_filtered = chat_copy.__class__.from_dataframe(df_filtered)
    else:
        # Si no tiene ese método, devolvemos el mismo objeto con atributo _df reemplazado
        if hasattr(chat_copy, "_df"):
            chat_copy._df = df_filtered
            chat_filtered = chat_copy
        else:
            raise AttributeError("No se puede reasignar df: la clase WhatsAppChat no lo permite y no tiene _df interno.")

    return chat_filtered


# -------------------------------
# Función para envolver gráficos en HTML
def wrap_plot(fig, title):
    fig.update_layout(width=1200, height=700)
    html_str = f"""
    <div style="text-align:center; margin:auto;">
        <h2>{title}</h2>
        {pio.to_html(fig, full_html=False, include_plotlyjs='cdn')}
    </div>
    """
    return html_str


# Función para generar nubes de palabras por usuario y devolver HTML
def wordclouds_html(df, stopwords, max_words=100):
    from wordcloud import WordCloud
    import base64
    from io import BytesIO
    import re

    html_parts = []
    df = filter_system_user(df)

    for user, group in df.groupby('username'):
        text = " ".join(group['message']).lower()
        words = re.findall(r'\b[a-záéíóúüñ]{3,}\b', text)
        filtered_words = set(w for w in words if w not in stopwords)

        if not filtered_words:
            continue

        wc_text = " ".join(filtered_words)
        wc = WordCloud(width=400, height=200, background_color='white', max_words=max_words).generate(wc_text)

        buffer = BytesIO()
        wc.to_image().save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        html_parts.append(f"""
        <div class="plot-container">
            <h3>{user}</h3>
            <img src="data:image/png;base64,{img_str}" />
        </div>
        """)

    if not html_parts:
        return "<p>No se encontraron palabras para generar nubes.</p>"

    return f"""
    <div class="plots-flex-container">
        {"".join(html_parts)}
    </div>
    """



# -------------------------------

def top_words_html(df, stopwords=None, cantidad=15):
    """
    Genera un HTML con las top palabras por usuario.
    Ignora palabras cortas y palabras 'flag' delimitadas por << >>.
    Las stopwords son opcionales.
    
    Args:
        df: DataFrame con columnas 'username' y 'message'
        stopwords: lista de palabras a ignorar (opcional)
        cantidad: número máximo de palabras por usuario
    
    Returns:
        HTML string con tabla estilizada
    """
    if stopwords is None:
        stopwords = set()
    else:
        stopwords = set(stopwords)
    
    user_word_counts = {}
    df = filter_system_user(df)
    for user, group in df.groupby('username'):
        text = " ".join(group['message']).lower()
        # Ignorar palabras flag <<...>>
        text = re.sub(r'<<.*?>>', '', text)
        words = re.findall(r'\b\w+\b', text)
        words = [w for w in words if w not in stopwords and len(w) > 2]
        if words:
            user_word_counts[user] = Counter(words).most_common(cantidad)
    
    if not user_word_counts:
        return "<p>No se encontraron palabras.</p>"
    
    # Construir filas HTML
    rows_html = ""
    for user, words in user_word_counts.items():
        word_spans = " ".join(
            f'<span style="font-weight:bold; font-size:1.2em; margin-right:8px;">{w} ({c})</span>'
            for w, c in words
        )
        rows_html += f"""
        <tr>
            <td style="font-weight:bold; font-size:1.1em; padding:10px; text-align:left; background-color:#f2f2f2;">{user}</td>
            <td style="padding:10px; text-align:left;">{word_spans}</td>
        </tr>
        """
    
    styled_html = f"""
    <div style="overflow-x:auto; margin:20px auto; max-width:90%;">
        <table style="border-collapse:collapse; width:100%; font-family:Arial, sans-serif; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background-color:#3498db; color:white; text-align:left;">
                    <th style="padding:10px;">Usuario</th>
                    <th style="padding:10px;">Palabras principales</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return styled_html

# ------------------------------------------------------------------------------------ Interactions network
'''
def interaction_network_html(df, user_map=USER_MAP):
    """
    Genera un grafo dirigido de interacciones entre usuarios usando menciones @numeros
    y el diccionario USER_MAP para mostrar nombres.
    """
    mentions = []

    for _, row in df.iterrows():
        from_name = user_map.get(row['username'], row['username'])
        # Captura todas las menciones que empiezan con @
        mentioned_items = re.findall(r'@[\w\d]+', row['message'])
        for m in mentioned_items:
            if m in user_map:
                to_name = user_map[m]
                mentions.append((from_name, to_name))

    if not mentions:
        return "<p>No se encontraron interacciones explícitas.</p>"

    # Crear grafo dirigido
    G = nx.DiGraph()
    for from_user, to_user in mentions:
        if G.has_edge(from_user, to_user):
            G[from_user][to_user]['weight'] += 1
        else:
            G.add_edge(from_user, to_user, weight=1)

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Edges con grosor proporcional a la cantidad de menciones
    edge_trace = []
    for from_user, to_user, data in G.edges(data=True):
        x0, y0 = pos[from_user]
        x1, y1 = pos[to_user]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=1 + data['weight'], color='#888'),
                hoverinfo='text',
                text=f"{from_user} → {to_user}: {data['weight']} menciones",
                mode='lines+markers',
            )
        )

    # Nodos
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} ({G.in_degree(node)} menciones recibidas)")
        node_size.append(10 + G.in_degree(node)*5)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition='top center',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=[G.in_degree(n) for n in G.nodes()],
            size=node_size,
            colorbar=dict(title='Menciones recibidas'),
            line_width=2
        )
    )

    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        title='Grafo de interacciones por menciones',
        showlegend=False,
        width=1000,
        height=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return wrap_plot(fig, "Grafo de interacciones")
'''

def group_events_plot_html(df, title="Intervenciones acumuladas por usuario y eventos"):

    df_plot = df.copy()
    df_plot['date'] = pd.to_datetime(df_plot['date'])

    # Detectar eventos
    def check_event(msg):
        if m := re.match(r'<<added:(.+)>>', msg):
            return "added", m.group(1).strip()
        elif m := re.match(r'<<exited:(.+)>>', msg):
            return "exited", m.group(1).strip()
        elif m := re.match(r'<<removed:(.+?)->(.+?)>>', msg):
            return "removed", f"{m.group(1).strip()} → {m.group(2).strip()}"
        return None, None

    df_plot[['event', 'event_user']] = df_plot['message'].apply(lambda x: pd.Series(check_event(x)))
    df_events = df_plot.dropna(subset=['event']).copy()

    # Contar mensajes por usuario
    df_messages = df_plot[df_plot['message'].notnull()].copy()
    df_messages['count'] = 1

    users = df_messages['username'].unique()
    #remove user <<system>> if exists
    users = [u for u in users if u != '<<system>>']
    fig = go.Figure()

    # Línea acumulada por usuario
    for user in users:
        df_user = df_messages[df_messages['username'] == user].copy()
        user_daily = df_user.groupby(df_user['date'].dt.date)['count'].sum().cumsum().reset_index()
        user_daily.rename(columns={'date':'date', 'count':'messages_cumulative'}, inplace=True)
        user_daily['date'] = pd.to_datetime(user_daily['date'])

        fig.add_trace(go.Scatter(
            x=user_daily['date'],
            y=user_daily['messages_cumulative'],
            mode='lines+markers',
            name=user,
            line=dict(width=2),
            marker=dict(size=5)
        ))

    # Determinar máximo acumulado para posicionar nombres de eventos
    max_y = df_messages.groupby(df_messages['date'].dt.date)['count'].sum().cumsum().max()

    # Líneas verticales para eventos con anotación
    color_map = {"added":"green", "exited":"red", "removed":"orange"}

    # Distribuir anotaciones verticalmente hacia abajo
    from collections import defaultdict
    event_positions = defaultdict(int)  # contador de eventos por fecha

    for _, row in df_events.iterrows():
        date_key = row['date'].date()
        event_positions[date_key] += 1
        vertical_offset = 1.0 - 0.05 * (event_positions[date_key]-1)  # bajar desde el borde superior

        fig.add_vline(
            x=row['date'],
            line=dict(color=color_map.get(row['event'], 'black'), width=2, dash="dash")
        )
        fig.add_annotation(
            x=row['date'],
            y=vertical_offset,  # posición descendente
            xref='x',
            yref='paper',
            text=row['event_user'],
            showarrow=False,
            font=dict(color=color_map.get(row['event'], 'black'), size=12),
            textangle=0,  # horizontal
            xanchor='left',
            yanchor='top'  # anclar arriba para que baje
        )

    fig.update_layout(
        width=1200, height=600,
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Mensajes acumulados",
        legend_title="Usuarios",
        yaxis=dict(showgrid=True, rangemode='tozero')
    )

    html_str = f"""
    <div style="text-align:center; margin:auto;">
        {pio.to_html(fig, full_html=False, include_plotlyjs='cdn')}
    </div>
    """
    return html_str





def group_interventions_plot(df, title="Intervenciones acumuladas por usuario"):

    # Funciones candidatas
    def linear(x, a, b):
        return a*x + b

    def logarithmic(x, a, b):
        return a * np.log(x+1) + b

    def sigmoid(x, L, k, x0, b):
        return L / (1 + np.exp(-k*(x-x0))) + b

    def exponential(x, a, b, c):
        return a * np.exp(b*x) + c

    def quadratic(x, a, b, c):
        return a*x**2 + b*x + c

    def cubic(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    def sqrt_func(x, a, b):
        return a * np.sqrt(x+1) + b

    def gompertz(x, a, b, c, d):
        return a * np.exp(-b * np.exp(-c*(x-d)))

    def loglogistic(x, a, b, c, d):
        return a / (1 + (b * np.exp(-c*(x-d))))

    # Diccionario de funciones para iterar
    CANDIDATE_FUNCS = {
        "lineal": linear,
        "logarítmica": logarithmic,
        "sigmoide": sigmoid,
        "exponencial": exponential,
        "cuadrática": quadratic,
        "cúbica": cubic,
        "raíz": sqrt_func,
        "gompertz": gompertz,
        "log-logística": loglogistic
    }

    df_plot = df.copy()
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    
    df_messages = df_plot[df_plot['message'].notnull()].copy()
    df_messages['count'] = 1
    
    df_cum = df_messages.groupby(df_messages['username']).apply(
        lambda x: x.groupby(x['date'].dt.date)['count'].sum().cumsum()
    ).reset_index()
    df_cum.rename(columns={'level_1':'date', 0:'cumulative'}, inplace=True)
    df_cum['date'] = pd.to_datetime(df_cum['date'])
    
    df_total = df_cum.groupby('date')['cumulative'].sum().reset_index()
    df_total['x'] = np.arange(len(df_total))
    
    xdata = df_total['x'].values
    ydata = df_total['cumulative'].values

    fits = {}
    for name, func in CANDIDATE_FUNCS.items():
        try:
            if func in [sigmoid, gompertz, loglogistic]:
                p0 = [max(ydata), 1, np.median(xdata), 0]
            elif func == exponential:
                p0 = [1, 0.1, min(ydata)]
            elif func == quadratic:
                p0 = [0.1, 0.1, min(ydata)]
            elif func == cubic:
                p0 = [0.01, 0.01, 0.01, min(ydata)]
            elif func == sqrt_func:
                p0 = [1, min(ydata)]
            else:
                p0 = None

            popt, _ = curve_fit(func, xdata, ydata, p0=p0, maxfev=5000)
            fits[name] = (func(xdata, *popt), popt)
        except:
            continue
    
    best_fit_name = None
    min_error = float('inf')
    for name, (yfit, _) in fits.items():
        error = np.mean((yfit - ydata)**2)
        if error < min_error:
            min_error = error
            best_fit_name = name
    
    # Mensajes descriptivos por función
    messages = {
        "lineal": "Tu grupo crece de manera constante en el tiempo 📏",
        "logarítmica": "Tu grupo crece rápido al inicio y luego se ralentiza 🐢",
        "sigmoide": "Tu grupo empieza lento, acelera y luego se estabiliza 📈",
        "exponencial": "Tu grupo crece explosivamente, ¡cada vez más rápido! 🔥",
        "cuadrática": "Tu grupo tiene un crecimiento que se acelera con el tiempo ⏩",
        "cúbica": "Tu grupo tiene un patrón complejo con aceleraciones y desaceleraciones 🎢",
        "raíz": "Tu grupo crece rápido al inicio y luego desacelera suavemente 🌱",
        "gompertz": "Tu grupo sigue una curva asimétrica, creciendo rápido al principio y luego aplanándose 📊",
        "log-logística": "Tu grupo tiene un crecimiento inicial lento, luego rápido, y finalmente se estabiliza ⚡"
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_total['date'], y=df_total['cumulative'],
                             mode='lines+markers', name='Intervenciones acumuladas'))

    if best_fit_name:
        fig.add_trace(go.Scatter(x=df_total['date'], y=fits[best_fit_name][0],
                                 mode='lines', name=f'Función ajustada ({best_fit_name})',
                                 line=dict(dash='dash', color='red')))
        message_text = messages.get(best_fit_name, f"Tu grupo es de tipo {best_fit_name}!")
    else:
        message_text = "No se pudo determinar un patrón de crecimiento claro 😅"

    fig.update_layout(
        width=1200, height=600,
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Intervenciones acumuladas",
        legend_title="Leyenda",
        yaxis=dict(showgrid=True, rangemode='tozero')
    )

    html_str = f"""
    <div style="text-align:center; margin:auto;">
        {pio.to_html(fig, full_html=False, include_plotlyjs='cdn')}
        <p style="font-size:16px; font-weight:bold;">{message_text}</p>
    </div>
    """
    return html_str


def filter_system_user(df):
    return df[df['username'] != '<<system>>'].copy()

    
    
def user_message_intervals_table_html(df, title="Intervalos entre mensajes por usuario"):

    df_plot = filter_system_user(df)
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    df_plot = df_plot.sort_values(['username', 'date'])

    # Calcular diferencias de tiempo entre mensajes consecutivos por usuario
    df_plot['prev_date'] = df_plot.groupby('username')['date'].shift(1)
    df_plot['interval'] = (df_plot['date'] - df_plot['prev_date'])
    
    # Filtrar solo intervalos válidos
    df_intervals = df_plot[df_plot['interval'].notnull()]

    # Promedio y máximo por usuario eliminando al usuario <<system>> si existe

    stats = df_intervals.groupby('username').agg(
        mean_interval=('interval', 'mean'),
        max_interval=('interval', 'max')
    ).reset_index()

    # Función para formatear timedelta en días, horas y minutos
    def format_timedelta(td):
        if pd.isna(td):
            return "0m"
        total_seconds = int(td.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, _ = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or (days == 0 and hours == 0):
            parts.append(f"{minutes}m")
        return " ".join(parts)

    stats['avg_interval'] = stats['mean_interval'].apply(format_timedelta)
    stats['max_interval'] = stats['max_interval'].apply(format_timedelta)

    # Fechas del intervalo máximo
    def get_max_interval_row(user_df):
        if user_df.empty:
            return pd.Series({'prev_date': pd.NaT, 'date': pd.NaT, 'interval': pd.Timedelta(0)})
        idx = user_df['interval'].idxmax()
        return user_df.loc[idx, ['prev_date', 'date', 'interval']]

    max_dates = df_intervals.groupby('username').apply(get_max_interval_row).reset_index()

    # Rango de fechas solo con día (sin hora)
    max_dates['max_interval_range'] = max_dates.apply(
        lambda row: f"{row['prev_date'].strftime('%Y-%m-%d') if pd.notna(row['prev_date']) else '-'} → {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '-'}",
        axis=1
    )

    # Unir todo
    stats = stats.merge(max_dates[['username', 'max_interval_range']], on='username', how='left')
    stats = stats[['username', 'avg_interval', 'max_interval', 'max_interval_range']]

    # Convertir a HTML
    html_table = stats.to_html(index=False, escape=False)

    return f"""
    <div style="text-align:center; margin:auto; width:90%;">
        <h3>{title}</h3>
        {html_table}
    </div>
    """

    
    



# -------------------------------
def top_emojis_html(df, cantidad=15):
    # Patrón para clusters de caracteres (grapheme clusters)
    df = filter_system_user(df)
    emoji_pattern = re.compile(r'\X', re.UNICODE)
    
    def extract_emojis(text):
        return [cluster for cluster in emoji_pattern.findall(text) if emoji.is_emoji(cluster)]

    # Diccionario para almacenar conteos por usuario
    user_emoji_counts = {}
    for user, group in df.groupby('username'):
        emojis = sum([extract_emojis(msg) for msg in group['message']], [])
        if emojis:  # Solo usuarios con al menos un emoji
            user_emoji_counts[user] = Counter(emojis).most_common(cantidad)

    if not user_emoji_counts:
        return "<p>No se encontraron emojis.</p>"

    # Construir DataFrame con cada emoji(count) en una celda
    rows = []
    for user, emojis in user_emoji_counts.items():
        row = {"Usuario": user}
        for i, (e, c) in enumerate(emojis, start=1):
            row[f"Emoji {i}"] = f"{e}({c})"
        rows.append(row)
    
    df_html = pd.DataFrame(rows).fillna('')  # Vacíos en vez de NaN
    return df_html.to_html(index=False, escape=False, border=1, justify='center')

# -------------------------------
# -------------------------------
# Función para análisis multimedia
def multimedia_html(df):
    def classify_multimedia(msg):
        if msg.startswith("<<") and msg.endswith(">>"):
            if "sticker" in msg:
                return "sticker"
            elif "audio" in msg:
                return "audio"
            elif "video" in msg:
                return "video"
            elif "imagen" in msg:
                return "image"
            elif "mensaje omitido" in msg:
                return "omitido"
            elif "deleted" in msg:
                return "mensaje borrado"
            else:
                return None

    df['multimedia_type'] = df['message'].apply(classify_multimedia)
    multimedia_counts = df[df['multimedia_type'].notnull()].groupby(['username','multimedia_type']).size().unstack(fill_value=0)

    fig_multimedia = go.Figure()
    for mtype in multimedia_counts.columns:
        fig_multimedia.add_trace(go.Bar(x=multimedia_counts.index, y=multimedia_counts[mtype], name=mtype))
    fig_multimedia.update_layout(title="Uso de multimedia por usuario", barmode='stack', width=1200, height=700,
                                xaxis_title="Usuario", yaxis_title="Cantidad")
    return wrap_plot(fig_multimedia, "Uso de multimedia por usuario")



# -------------------------------
# Función para wraps de gráficos
def wrap_plot(fig, title):
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# -------------------------------
# Aperturas y cierres de conversación
def compute_conversation_open_close(df, threshold_hours=8):
    df = df.sort_values('date').copy()
    threshold = pd.Timedelta(hours=threshold_hours)
    df['time_diff'] = df['date'].diff()
    df['long_break'] = df['time_diff'] > threshold
    df['open_convo'] = df['long_break']
    df['close_convo'] = df['time_diff'].shift(-1) > threshold

    open_counts = df[df['open_convo']].groupby('username').size().sort_values(ascending=False)
    close_counts = df[df['close_convo']].groupby('username').size().sort_values(ascending=False)

    fig_open = go.Figure([go.Bar(x=open_counts.index, y=open_counts.values)])
    fig_open.update_layout(title="Aperturas de conversación por usuario", width=1200, height=700)

    fig_close = go.Figure([go.Bar(x=close_counts.index, y=close_counts.values)])
    fig_close.update_layout(title="Cierres de conversación por usuario", width=1200, height=700)

    return wrap_plot(fig_open, "Aperturas de conversación"), wrap_plot(fig_close, "Cierres de conversación")

# -------------------------------
# Estadísticas de emojis

def emoji_statistics(df):
    emoji_pattern = re.compile(r'[^\w\s,]')
    df = df.copy()
    df['emoji_count'] = df['message'].apply(lambda x: len(emoji_pattern.findall(x)))
    stats = df.groupby('username').agg(
        total_emojis=('emoji_count', 'sum'),
        messages_with_emojis=('emoji_count', lambda x: (x>0).sum()),
        total_messages=('message', 'count')
    )
    stats['pct_messages_with_emojis'] = 100*stats['messages_with_emojis']/stats['total_messages']

    fig_emoji = go.Figure([go.Bar(x=stats.index, y=stats['total_emojis'])])
    fig_emoji.update_layout(title="Total de emojis por usuario", width=1200, height=700)

    return wrap_plot(fig_emoji, "Total de emojis por usuario")

# -------------------------------
# Mensajes más largos
def top_long_messages(df, metadata=False):
    df = filter_system_user(df)
    if 'msg_length' not in df.columns:
        df['msg_length'] = df['message'].str.len()
    top_messages = df.loc[df.groupby('username')['msg_length'].idxmax(), ['username','message','date'] + (['msg_length'] if metadata else [])]
    return chat_table_html(top_messages, user_col='username', message_col='message', date_col='date', metadata=metadata)



