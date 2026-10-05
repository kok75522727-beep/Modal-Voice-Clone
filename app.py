import modal
from typing import Dict

# Modal ဆာဗာမှာ အသံထုတ်ဖို့ လိုအပ်တဲ့ စနစ်တွေ ကြိုတင်သွင်းခြင်း
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", 
    "azure-cognitiveservices-speech",
    "gradio_client",
    "pydub"
)

app = modal.App("voice-clone-api")

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_audio(data: Dict):
    text = data.get("text", "")
    voice = data.get("voice", "Nilar")
    
    return {
        "status": "Success", 
        "message": "Modal ဆာဗာမှ အောင်မြင်စွာ လက်ခံရရှိပါသည်!",
        "received_text": text,
        "selected_voice": voice
    }
