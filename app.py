import modal
from fastapi import FastAPI, Request

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
async def generate_audio(request: Request):
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", "Nilar")
    
    # ဒီနေရာမှာ နောက်ပိုင်း Website ကနေ ပို့လိုက်တဲ့ စာတွေကို 
    # တကယ် အသံပြောင်းပေးမယ့် အပိုင်း ဝင်လာပါမယ်။
    return {
        "status": "Success", 
        "message": "Modal ဆာဗာမှ အောင်မြင်စွာ လက်ခံရရှိပါသည်!",
        "received_text": text,
        "selected_voice": voice
    }
