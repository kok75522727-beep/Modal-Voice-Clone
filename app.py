import modal

# AI Model များကို ဆာဗာတည်ဆောက်ချိန်တွင် ကြိုတင်ဒေါင်းလုဒ်ဆွဲထားမည်
def download_models():
    from TTS.api import TTS
    # အဆင့်မြင့် Voice Conversion Model ကို အသုံးပြုထားပါသည်
    TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False)

# မြန်နှုန်းမြင့် T4 GPU နှင့် လိုအပ်သော AI စနစ်များ တပ်ဆင်ခြင်း
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "espeak-ng", "libsndfile1")
    .run_commands(
        "pip install --upgrade pip",
        "pip install wheel packaging",
        "pip install torch torchaudio",
        "pip install fastapi[standard] python-multipart pydub",
        "pip install TTS==0.22.0",
        "pip install setuptools==69.5.1"
    )
    .run_function(download_models)
)

app = modal.App("oneteam-voice-clone-pro")

# 🚀 GPU T4 ကို အပြည့်အဝ အသုံးပြု၍ အချိန်တိုအတွင်း လုပ်ဆောင်ပါမည်
@app.function(image=image, gpu="T4", timeout=1200)
@modal.asgi_app()
def my_voice_clone_api():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import Response
    import tempfile
    import os
    from pydub import AudioSegment
    import torch
    from TTS.api import TTS

    web_app = FastAPI()

    # GPU စနစ်ဖြင့် AI အင်ဂျင်ကို မောင်းနှင်မည်
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

            # အသံဖိုင်များကို AI မှ အကောင်းဆုံး ခွဲခြမ်းစိတ်ဖြာနိုင်ရန် 30s အပိုင်းကြီးများ ပိုင်းခြားမည်
            audio = AudioSegment.from_file(base_path)
            chunk_length_ms = 30000
            chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            combined_audio = AudioSegment.empty()

            for chunk in chunks:
                chunk_path = tempfile.mktemp(suffix=".wav")
                chunk.export(chunk_path, format="wav")
                
                chunk_out_path = tempfile.mktemp(suffix=".wav")
                
                # Zero-shot AI အင်ဂျင်ဖြင့် အသံအရောင်နှင့် လေယူလေသိမ်းကို အတိအကျ ကူးယူခြင်း
                tts.voice_conversion_to_file(source_wav=chunk_path, target_wav=ref_path, file_path=chunk_out_path)
                
                converted_chunk = AudioSegment.from_file(chunk_out_path)
                combined_audio += converted_chunk
                
                os.unlink(chunk_path)
                os.unlink(chunk_out_path)

            final_out_path = tempfile.mktemp(suffix=".wav")
            
            # အသံပိုမိုကြည်လင် သဘာဝကျစေရန် Audio Normalize အနည်းငယ် လုပ်ပေးမည်
            combined_audio = combined_audio.normalize()
            combined_audio.export(final_out_path, format="wav")
            
            with open(final_out_path, 'rb') as f:
                final_bytes = f.read()
                
            os.unlink(base_path)
            os.unlink(ref_path)
            os.unlink(final_out_path)

            return Response(content=final_bytes, media_type="audio/wav")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")
            
    return web_app
