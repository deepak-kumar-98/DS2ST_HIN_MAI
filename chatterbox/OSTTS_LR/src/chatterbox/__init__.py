try:
    from importlib.metadata import version
    __version__ = version("chatterbox-tts")
except Exception:
    # Fallback when package not installed
    __version__ = "0.1.4"


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES