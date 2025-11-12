import os
import pandas as pd
from formatting_utils import generate_collapsible_sections_modern, generate_index_modern, get_collapsible_script_modern
from lda_utils import lda_analysis
import plotly.graph_objects as go
from utils import *
from sentiment_utils import *
from normalize_utils import *  # si tienes función de normalización


BASE_DIR = r"C:\Users\carlo\Documents\chatanalizer\\"
INPUT_DIR = os.path.join(BASE_DIR, "input")
CHATS_DIR = os.path.join(BASE_DIR, "chats")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_phone_map(csv_path):
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return dict(zip(df['tel'], df['name']))


def clean_message(msg):
    """Elimina caracteres invisibles y extraños, mantiene texto, emojis y Unicode estándar"""
    if not isinstance(msg, str):
        return ""
    invisibles = [
        '\u200e', '\u200f', '\u202f', '\u2060', '\ufeff', '\u200b'
    ]
    msg = ''.join(c for c in msg if c not in invisibles and (c.isprintable() or ord(c) > 127))
    return msg.strip()


def main(chat_filename="out.txt"):
    chat_path = os.path.join(CHATS_DIR, chat_filename)
    phone_map_path = os.path.join(INPUT_DIR, "phone_map.csv")
    phone_map = load_phone_map(phone_map_path)

    print(f"Cargando chat desde: {chat_path}")
    
    
    
    chat, df = load_chat(chat_path)  
    
        
    # TODO: eliminar filas si hay mucho tiempo entre mensajes y es mensaje de creacion
    # Eliminar las dos primeras filas
    df = df.iloc[2:].reset_index(drop=True)    
    
    # Reemplazar teléfonos por nombres si existen
    df['username'] = df['username'].apply(lambda x: phone_map.get(x, x))
    # Limpiar mensajes
    df['message'] = df['message'].apply(clean_message)
    #print unique users
    sys_print_debug(msg=f"Unique users: {df['username'].nunique()} - {df['username'].unique()}")




    
    fb = FigureBuilder(chat=remove_system_from_wa_chat(chat))  # tu clase de gráficas

    html_strings = {}

    # === Analisis temporal de intervenciones ===
    print('Generando gráficos de intervenciones...')
    linechart_configs = [
        {}, {"cumulative": True}, {"cumulative": True, "all_users": True},
        {"msg_length": True, "cumulative": True}, {"date_mode": "hour", "xlabel": "Hora"},
        {"date_mode": "weekday", "xlabel": "Día de la semana"}, {"date_mode": "month", "xlabel": "Mes"}
    ]
    titles = [
        "Intervenciones por usuario", "Intervenciones acumuladas por usuario",
        "Intervenciones acumuladas (todos los usuarios)", "Caracteres enviados (cumulativo)",
        "Intervenciones por hora", "Intervenciones por día de la semana", "Intervenciones por mes"
    ]
    for cfg, title in zip(linechart_configs, titles):
        fig = fb.user_interventions_count_linechart(**cfg)
        html_strings['wrap_plot'] = wrap_plot(fig, title)
        
    # === Intervalos entre mensajes de usuarios ===
    print('Generando Intervalos entre mensajes de usuarios...')
    html_strings['user_message_intervals_table_html'] = user_message_intervals_table_html(df)


    # === Eventos de grupo: añadidos, salidas, eliminaciones ===
    print('Generando gráfico de eventos de grupo...')
    html_strings['group_events_plot_html'] = group_events_plot_html(df)
    

    # === Analisis de texto ===
    stopwords = {"que", "los", "del", "las", "por", "para", "con", "una", "como",
                 "más", "pero", "sus", "esta", "este", "voy", "eso"}
    html_strings['top_words_html'] = top_words_html(df, stopwords)
    html_strings['wordclouds_html'] = wordclouds_html(df, stopwords)

    # === Analisis de emojis ===
    html_strings['top_emojis_html'] = top_emojis_html(df)

    # === Multimedia ===
    html_strings['multimedia_html'] = multimedia_html(df)

    # === Longitud de mensajes ===
    if 'msg_length' not in df.columns:
        df['msg_length'] = df['message'].str.len()
    html_strings['long_messages'] = wrap_plot(fb.user_msg_length_boxplot(),
                                  "Boxplot de longitud de mensajes")

    # Boxplot sin los 5 mensajes más largos por usuario
    #filtered_data = pd.concat([
    #    group.nsmallest(len(group)-5, 'msg_length') if len(group) > 5 else group
    #    for _, group in df.groupby('username')
    #])
    #fb_filtered = FigureBuilder(chat=WhatsAppChat(filtered_data))
    #html_strings[] = wrap_plot(fb_filtered.user_msg_length_boxplot(),
    #                              "Boxplot sin los 5 mensajes más largos por usuario"))

    # === Respuestas entre usuarios ===
    html_strings['Heatmap respuestas'] = wrap_plot(fb.user_message_responses_heatmap(),
                                  "Heatmap de respuestas entre usuarios")
    html_strings['Flujo de respuestas'] = wrap_plot(fb.user_message_responses_flow(),
                                  "Flujo de respuestas entre usuarios")

    # === Caracteres promedio vs intervenciones ===
    counts_interv = get_interventions_count(chat=chat, date_mode='date', msg_length=False, cumulative=False)
    counts_len = get_interventions_count(chat=chat, date_mode='date', msg_length=True, cumulative=False)
    counts_len = pd.DataFrame(counts_len.unstack(), columns=['num_characters'])
    counts_interv = pd.DataFrame(counts_interv.unstack(), columns=['num_interventions'])
    counts = counts_len.merge(counts_interv, left_index=True, right_index=True)
    counts = counts[counts['num_interventions'] != 0].reset_index()
    counts['avg_characters'] = counts['num_characters'] / counts['num_interventions']

    traces = []
    for username in fb.usernames:
        counts_user = counts[counts['username'] == username]
        traces.append(go.Histogram2dContour(
            contours={'coloring': 'none'},
            x=counts_user['num_interventions'],
            y=counts_user['avg_characters'],
            name=username,
            showlegend=True,
            line={'color': fb.user_color_mapping.get(username, "#000000")},
            nbinsx=10, nbinsy=20
        ))
    fig_hist2d = go.Figure(data=traces)
    fig_hist2d.update_layout(title='Avg characters sent per day vs Interventions per day',
                             xaxis_title='Num interventions',
                             yaxis_title='Avg characters',
                             width=800, height=600)
    html_strings['Caracteres promedio vs intervenciones'] = wrap_plot(fig_hist2d, "Caracteres promedio vs intervenciones")



    # === Análisis adicional, red de interacciones, sentimientos ===
                                                                      
    html_strings['top_long_messages'] = top_long_messages(df)
    html_strings['compute_conversation_open_close'] = compute_conversation_open_close(df)
    html_strings['emoji_statistics'] = emoji_statistics(df)
                                                                      
    #html_strings[] = interaction_network_html(df))
    res_df = sentiment_analysis_complete(chat_path, df)

    # === Generar gráficos HTML ===
    html_strings['polarity_statistics'] = generate_polarity_statistics(res_df)
    html_strings['emotion_statistics'] = generate_emotion_statistics(res_df)
    html_strings['emotion_over_time'] = generate_emotion_over_time(res_df)
    html_strings['top_emotions'] = generate_top_emotions(res_df)
    html_strings['user_emotion_pies'] = generate_user_emotion_pies(res_df)
    html_strings['top_user_per_emotion'] = generate_top_user_per_emotion(res_df)


    # Después de análisis de texto y antes de guardar
    print("Generando análisis de tópicos (LDA)...")
    html_strings['lda_topics'] = lda_analysis(df, num_topics=10)


    # Convertir todo a string
    #TODO: CHANGE ON FUNCTIONS NOT HERE
    # Convertir DataFrames a HTML dentro del diccionario
    for key, value in html_strings.items():
        if isinstance(value, pd.DataFrame):
            html_strings[key] = value.to_html(index=False, escape=False)
        elif isinstance(value, list):
            # Unir los elementos de la lista sin corchetes ni comas
            html_strings[key] = "".join(value)
        else:
            html_strings[key] = str(value)
        

    chat_stem = os.path.splitext(chat_filename)[0]
    html_report_path = os.path.join(OUTPUT_DIR, f"{chat_stem}-report.html")

    # Uso en tu html_report
    section_titles = html_strings.keys()  # reemplazar por tus títulos
    sections_html = generate_collapsible_sections_modern(html_strings)
    index_html = generate_index_modern(section_titles)
    collapsible_script = get_collapsible_script_modern()

    html_report = f"""
    <html>
    <head>
    <meta charset="UTF-8">
    <title>WhatsApp Chat Report</title>
    <style>
    body {{ font-family: Arial, sans-serif; text-align: center; background-color: #f9f9f9; color: #333; margin:20px; }}
    h1 {{ color: #2c3e50; margin-bottom:30px; }}
    table {{ border-collapse: collapse; margin:0 auto 30px auto; width:80%; max-width:1000px; box-shadow:0 2px 5px rgba(0,0,0,0.1); }}
    th, td {{ border:1px solid #ddd; padding:8px 12px; text-align:center; }}
    th {{ background-color:#3498db; color:white; font-size:16px; }}
    tr:nth-child(even) {{ background-color:#f2f2f2; }}
    tr:hover {{ background-color:#e0f7fa; }}
    .plot-container {{ margin:20px auto; }}
    </style>
    </head>
    <body>
    <h1>WhatsApp {txt_file.removesuffix(".txt")} Chat Report</h1>
    {index_html}
    {sections_html}
    {collapsible_script}
    </body>
    </html>
    """
    
    
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"Reporte generado: {html_report_path}")


if __name__ == "__main__":
    # 1️⃣ Normalizar todos los zips en la carpeta input
    for zip_file in os.listdir(INPUT_DIR):
        if zip_file.lower().endswith(".zip"):
            zip_path = os.path.join(INPUT_DIR, zip_file)
            print(f"Normalizando {zip_file}...")
            normalize_whatsapp_chats(zip_file=zip_path)
            print(f"Normalizado")


    # 2️⃣ Procesar todos los TXT normalizados
    for txt_file in os.listdir(CHATS_DIR):
        if txt_file.lower().endswith(".txt"):
            output_path = os.path.join(OUTPUT_DIR, txt_file)
            print(f"Analizando {txt_file}...")
            main(os.path.join(CHATS_DIR, txt_file))