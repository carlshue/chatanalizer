import pandas as pd
import hashlib
import colorsys


def get_collapsible_script():
    """Script básico para secciones colapsables."""
    return """
    <script>
    var coll = document.getElementsByClassName("collapsible");
    for (var i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.display === "block") {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        });
    }
    </script>
    """


def chat_table_html(df, user_col='username', message_col='message', date_col='date',
                    max_height=400, metadata=False):
    """
    Genera un HTML tipo chat estilo WhatsApp desde un DataFrame.
    Cada usuario recibe un color pastel fijo. Metadatos opcionales se muestran sutilmente a la derecha.

    Args:
        df: DataFrame con columnas de usuario, mensaje, fecha y opcionales.
        user_col: nombre de la columna de usuario.
        message_col: nombre de la columna de mensaje.
        date_col: nombre de la columna de fecha.
        max_height: altura máxima del contenedor de chat en px.
        metadata: si True, se muestran metadatos adicionales.

    Returns:
        html_str: string con el HTML del chat.
    """
    df = df.dropna(subset=[user_col, message_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(by=date_col)

    # Genera un color pastel único por usuario
    def pastel_color(username):
        h = int(hashlib.md5(username.encode()).hexdigest()[:6], 16) % 360
        r, g, b = colorsys.hls_to_rgb(h / 360, 0.7, 0.6)
        return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'

    colors = {user: pastel_color(user) for user in df[user_col].unique()}
    meta_cols = [c for c in df.columns if c not in [user_col, message_col, date_col]]

    chat_html = f"""
    <div style='max-height:{max_height}px; overflow-y:auto; padding:15px; 
                border:1px solid #ccc; font-family:Arial,sans-serif; display:flex; 
                flex-direction:column; gap:8px;'>
    """

    for _, row in df.iterrows():
        user = row[user_col]
        msg = row[message_col]
        date_str = row[date_col].strftime('%Y-%m-%d %H:%M') if pd.notna(row[date_col]) else ""
        color = colors[user]

        meta_html = ""
        if metadata:
            for c in meta_cols:
                val = row[c]
                if pd.notna(val):
                    meta_html += f"""
                    <div style='font-size:0.7em; color:rgba(50,50,50,0.7);
                                text-shadow:1px 1px 2px rgba(0,0,0,0.1); margin-top:2px;'>
                        {c}: {val}
                    </div>"""

        chat_html += f"""
        <div style='display:flex; flex-direction:column; max-width:80%;'>
            <div style='font-size:0.85em; font-weight:bold; color:#333; margin-bottom:2px;'>{user} — {date_str}</div>
            <div style='display:flex; gap:10px; align-items:flex-start;'>
                <div style='background-color:{color}; color:#000; padding:10px 14px; border-radius:20px;
                            word-wrap:break-word; box-shadow:1px 1px 4px rgba(0,0,0,0.1);'>
                    {msg}
                </div>
                <div style='flex-shrink:0;'>{meta_html}</div>
            </div>
        </div>
        """

    chat_html += "</div>"
    return chat_html


def generate_index_modern(sections):
    """Genera un índice flotante con estilo moderno y scrollspy visual."""
    index_html = """
    <div id="index" style="position:fixed; top:60px; right:20px; width:220px;
        background-color:#f9f9f9; padding:15px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.15);">
        <h3 style="margin-top:0; color:#2c3e50;">Índice</h3>
        <ul style="list-style:none; padding-left:0;">
    """
    for i, section in enumerate(sections):
        index_html += f'<li><a href="#section{i}" class="index-link" style="text-decoration:none; color:#3498db;">{section}</a></li>'
    index_html += "</ul></div>"
    return index_html


def generate_collapsible_sections_modern(html_dict):
    """Genera secciones colapsables con transición suave y estilo moderno."""
    sections_html = ''
    for i, (title, content) in enumerate(html_dict.items()):
        sections_html += f"""
        <div style="margin-bottom:15px;">
            <button class="collapsible" style="background-color:#3498db;color:white;padding:12px;width:80%;
                    border:none;text-align:left;font-size:16px;font-weight:bold;border-radius:5px;cursor:pointer;">
                {title}
            </button>
            <div class="content" id="section{i}" style="padding:0 18px; max-height:0; overflow:hidden;
                    transition:max-height 0.3s ease-out; background-color:#f9f9f9; border-radius:0 0 5px 5px;">
                {content}
            </div>
        </div>
        """
    return sections_html


def get_collapsible_script_modern():
    """Script moderno para secciones colapsables y scrollspy de índice."""
    return """
    <script>
    // Colapsables
    var coll = document.getElementsByClassName("collapsible");
    for (var i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            content.style.maxHeight = content.style.maxHeight ? null : content.scrollHeight + "px";
        });
    }

    // ScrollSpy
    window.addEventListener('scroll', function() {
        var sections = document.querySelectorAll('.content');
        var links = document.querySelectorAll('.index-link');
        var scrollPos = window.scrollY || window.pageYOffset;

        sections.forEach(function(section, i) {
            if (section.offsetTop <= scrollPos + 100 && (section.offsetTop + section.offsetHeight) > scrollPos + 100) {
                links.forEach(l => { l.style.fontWeight='normal'; l.style.color='#3498db'; });
                links[i].style.fontWeight = 'bold';
                links[i].style.color = '#e74c3c';
            }
        });
    });
    </script>
    """
