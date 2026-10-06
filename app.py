import modal

# ၁။ AI Model ကို ဆာဗာစတင်ချိန်မှာ ကြိုတင်ဒေါင်းလုဒ်ဆွဲထားမည့် Function
def download_models():
    from TTS.api import TTS
    TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False)

# ၂။ တကယ့် AI Model ကို ကိုယ်ပိုင်ဆာဗာမှာ သွင်းမည့်အပိုင်း (run_function ဖြင့် တွဲထည့်ထားသည်)
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]", 
        "python-multipart",
        "pydub",
        "TTS", 
        "torch",
        "torchaudio"
    )
    .run_function(download_models)
)

app = modal.App("oneteam-voice-clone-pro")

# ၃။ GPU ('T4') ကိုသုံးပြီး အသံအမြန်ပြောင်းမည့် API
@app.function(image=image, gpu="T4", timeout=600)
@modal.asgi_app()
def my_voice_clone_api():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import Response
    import tempfile
    import os
    from pydub import AudioSegment
    from TTS.api import TTS
    import torch

    web_app = FastAPI()

    # Model ကို GPU ပေါ်တင်ထားမည် (အမြန်ပြောင်းနိုင်ရန်)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)

    @web_app.post("/")
    async def process_audio(base_audio: UploadFile = File(...), ref_audio: UploadFile = File(...)):
        try:
            base_bytes = await base_audio.read()
            ref_bytes = await ref_audio.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                tmp_ref.write(ref_bytes)
                ref_path = tmp_ref.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_base:
                tmp_base.write(base_bytes)
                base_path = tmp_base.name

            # Memory မပြည့်အောင် ၁၅ စက္ကန့်စီ ပိုင်းပြီး GPU နဲ့ ပြောင်းမည်
            audio = AudioSegment.from_file(base_path)
            chunk_length_ms = 15000
            chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            combined_audio = AudioSegment.empty()

            for chunk in chunks:
                chunk_path = tempfile.mktemp(suffix=".wav")
                chunk.export(chunk_path, format="wav")
                
                chunk_out_path = tempfile.mktemp(suffix=".wav")
                
                # 🚀 ကိုယ်ပိုင် GPU ဖြင့် အသံပွားခြင်း
                tts.voice_conversion_to_file(source_wav=chunk_path, target_wav=ref_path, file_path=chunk_out_path)
                
                converted_chunk = AudioSegment.from_file(chunk_out_path)
                combined_audio += converted_chunk
                
                os.unlink(chunk_path)
                os.unlink(chunk_out_path)

            final_out_path = tempfile.mktemp(suffix=".wav")
            combined_audio.export(final_out_path, format="wav")
            
            with open(final_out_path, 'rb') as f:
                final_bytes = f.read()
                
            os.unlink(base_path)
            os.unlink(ref_path)
            os.unlink(final_out_path)

            return Response(content=final_bytes, media_type="audio/wav")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ကိုယ်ပိုင် AI Error: {str(e)}")
            
    return web_app
