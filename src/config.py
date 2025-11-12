
CONFIG = {
    "VERBOSE": False,  # activar mensajes detallados
    "SENTIMENT_ANALYSIS": False,  # activar análisis de sentimiento
    "REMOVE_STRANGE_CHARS_WHILE_NORMALIZING": True # eliminar caracteres extraños al normalizar
}



def get_config():
    """Devuelve la configuración de depuración."""
    return CONFIG
    
def sys_print_debug(msg: str, level="DEBUG", conditions: list = None):
    if get_config().get("VERBOSE") is False:
        return
    
    if not conditions:
        print(f'[{level}]: {msg}')
        
    elif all(conditions):
        print(f'[{level}]: {msg}')
