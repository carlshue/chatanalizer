# normalizar_whatsapp.py
import re
import pathlib
import zipfile
import csv
from datetime import datetime

from config import *

VERBOSE = get_config().get("VERBOSE", False)
REMOVE_STRANGE_CHARS_WHILE_NORMALIZING = get_config().get("REMOVE_STRANGE_CHARS_WHILE_NORMALIZING", True)

def normalize_whatsapp_chats(zip_file=None, txt_file=None):
    INPUT_FOLDER = pathlib.Path("input")
    OUTPUT_FOLDER = pathlib.Path("chats")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # --- Cargar CSV de teléfonos si existe ---
    phone_map = {}
    csv_path = INPUT_FOLDER / "phone_map.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                phone_map[row['tel']] = row['name']

    # --- Patrones ---
    pat_brackets = re.compile(r'^\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?::\d{2})?)\]\s')
    pat_dash     = re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?\s?- ')

    def is_new_msg(line: str) -> bool:
        return bool(pat_brackets.match(line) or pat_dash.match(line))

    def normalize_datetime(date_str: str, time_str: str) -> str:
        for fmt in ("%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M:%S"):
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", fmt)
                break
            except ValueError:
                continue
        else:
            return f"{date_str} {time_str}"
        try:
            return dt.strftime("%m/%d/%y, %-I:%M %p")  # Linux/Mac
        except ValueError:
            return dt.strftime("%m/%d/%y, %#I:%M %p")  # Windows

    def replace_phones(msg):
        for phone, name in phone_map.items():
            msg = msg.replace(phone, name)
        return msg


    def clean_message(user, msg):

        # Documentos, stickers, audio, video, mensaje omitido
        msg = re.sub(
            r'[\u200e](sticker|audio|imagen|video|documento|mensaje|GIF|Tarjeta de contacto)\s+omitid[oa]?',
            lambda m: f"<<doc:{m.group(1).lower()}>>",
            msg,
            flags=re.IGNORECASE
        )

        # Añadidos (expande múltiples nombres)
            
       #def expand_added(m):
       #    prefix = m.group(1).strip()   # Esto es "12/25/23, 1:32 AM - difusión fiesta:"
       #    quien = m.group(2).strip()    # Esto es "Daniel🐆"
       #    personas_raw = m.group(3).strip(',')

       #    # Separar por comas y " y "
       #    personas = re.split(r',\s*| y ', personas_raw)
       #    personas = [p.strip() for p in personas if p.strip()]

       #    # Generar un flag por persona añadida, repitiendo el prefijo en cada línea
       #    return "\n".join(f"{prefix} <<added:{quien} -> {p}>>" for p in personas)


        msg = re.sub(
            r'[\u200e](.+?) añadió a (.+?)\.',
            lambda m: f"<<added:{m.group(1).strip()} -> {m.group(2).strip()}>>",
            msg,
            flags=re.IGNORECASE | re.MULTILINE
        )


        # Salidas
        msg = re.sub(
            r'[\u200e](.+?) salió del grupo\.?',
            lambda m: f"<<exited:{m.group(1).strip()}>>",
            msg,
            flags=re.IGNORECASE
        )

                
        msg = re.sub(
            r'\u200e[a|A]ñadiste a (.+?)\.',
            lambda m: f"<<added:{m.group(1).strip()}>>",
            msg,
            flags=re.IGNORECASE
        )
        

        # Eliminaciones de mensajes
        msg = re.sub(
            r'[\u200e](.+?) eliminó este mensaje\.?',
            lambda m: f"<<deleted>>",
            msg,
            flags=re.IGNORECASE
        )

        msg = re.sub(
            r'[\u200e](.+?) eliminó a ([^\.]+)\.',
            lambda m: f"<<removed:{m.group(1).strip()}->{m.group(2).strip()}>>",
            msg,
            flags=re.IGNORECASE
        )
        
        # Mensajes normales (bandera)
        if not any(msg.startswith(prefix) for prefix in ("<<doc:", "<<added:", "<<exited:", "<<deleted:", "<<removed:")):
            msg = re.sub(
                r'‎(?!\[\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?\])(.+)',
                lambda m: f"<<flag:{m.group(1).strip()}>>",
                msg
            )

        msg = replace_phones(msg)
        msg = ''.join(c for c in msg if c.isprintable() or c in "\n\t" or ord(c) > 127)
        return msg.strip()


    def process_text_file(txt_path, output_name=None):
        group_name = output_name
        text = txt_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r'^WhatsApp Chat -\s*', '', text, flags=re.IGNORECASE)
        lines = text.split("\n")[2:]  # eliminar primera línea

        out = []
        curr = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('‎'):                # Remove invalid character at start
                line = line[1:]
                
            if re.search(r'[\u200e]+Tú:', line):         # Remove Tú lines TODO: Expand to other languages
                continue
                
            if f" {group_name}: " in line:                # Remove group system general messages
                continue  

            if is_new_msg(line):
                # Si hay mensaje acumulado multilinea, añadirlo
                if curr:
                    out.append(curr)
                    curr = ""

                m = pat_brackets.match(line)
                if m:
                    date_str, time_str = m.group(1), m.group(2)
                    dt = normalize_datetime(date_str, time_str)
                    rest = line[m.end():].strip()
                else:
                    try:
                        date_str, rest = line.split(",", 1)
                        time_str, rest = rest.strip().split("-", 1)
                        dt = normalize_datetime(date_str.strip(), time_str.strip())
                    except Exception:
                        dt = ""
                        rest = line

                if ": " in rest:
                    user, msg = rest.split(": ", 1)
                else:
                    user, msg = "<<system>>", rest

                msg = clean_message(dt, msg.strip())

                if msg.startswith("<<added:"):
                    content = msg[len("<<added:"):msg.index(">>")].strip()
                    if "->" in content:
                        adder, added_str = content.split("->", 1)
                        adder = adder.strip()
                        added_str = added_str.strip()
                        added_str = added_str.replace(" y ", ", ")
                        users = [u.strip() for u in added_str.split(",") if u.strip()]
                        for u in users:
                            out.append(f"{dt} - <<system>>: <<added:{adder} -> {u}>>")
                    else:
                        out.append(f"{dt} - <<system>>: {msg}")
                else:
                    # Mensaje normal de una sola línea
                    out.append(f"{dt} - {user}: {msg}")
            else:
                # Mensaje multilinea
                curr += " " + line

        # Si queda algún mensaje multilinea al final
        if curr:
            out.append(curr)

        clean_name = re.sub(r'^WhatsApp Chat -\s*', '', txt_path.stem, flags=re.IGNORECASE)
        dst_name = output_name + ".txt" if output_name else clean_name + ".txt"
        dst = OUTPUT_FOLDER / dst_name
        dst.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"Hecho: {dst}")

    # --- Procesar ZIP específico ---
    if zip_file:
        zip_path = pathlib.Path(zip_file)
        zip_stem_clean = re.sub(r'^WhatsApp Chat -\s*', '', zip_path.stem, flags=re.IGNORECASE)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for txt_name in zf.namelist():
                if txt_name.lower().endswith(".txt"):
                    with zf.open(txt_name) as f:
                        tmp_path = OUTPUT_FOLDER / txt_name
                        tmp_path.write_bytes(f.read())
                        process_text_file(tmp_path, output_name=zip_stem_clean)
                        tmp_path.unlink()                                                                       # REMOVE TEMPORARY FILE
        return

    # --- Procesar TXT específico ---
    if txt_file:
        process_text_file(pathlib.Path(txt_file))
        return

    # --- Procesar todos los TXT sueltos si no se pasa nada ---
    for txt_path in INPUT_FOLDER.glob("*.txt"):
        output_path = OUTPUT_FOLDER / txt_path.name
        if output_path.exists():
            print(f"El archivo {output_path.name} ya existe, pasando al siguiente...")
            continue
        process_text_file(txt_path)

    print("Todos los archivos procesados.")
