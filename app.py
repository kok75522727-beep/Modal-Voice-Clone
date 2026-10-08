import modal

def download_models():
    from huggingface_hub import snapshot_download
    snapshot_download("openbmb/VoxCPM2")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .run_commands(
        "pip install --upgrade pip",
        "pip install torch torchaudio",
        "pip install soundfile librosa fastapi[standard] python-multipart pydub numpy huggingface_hub",
        "pip install voxcpm ninja"
    )
    .run_function(download_models)
)

app = modal.App("oneteam-voice-clone-pro")

# 🚀 အလုပ်ပိုမြန်စေရန် T4 အစား L4 GPU ကို အသုံးပြုထားပါသည်
@app.function(image=image, gpu="L4", timeout=1500)
@modal.asgi_app()
def my_voice_clone_api():
    from fastapi import FastAPI, UploadFile, File, HTTPException, Form
    from fastapi.responses import Response
    import tempfile
    import os
    import numpy as np
    import soundfile as sf
    import re
    from voxcpm import VoxCPM

    web_app = FastAPI()

    print("Loading Official VoxCPM2 Model...")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    print("Model Loaded Successfully!")

    def split_text_into_chunks(text, max_length=120):
        text = text.replace("\n", " ")
        raw_chunks = re.split(r'(?<=[။၊])\s*', text)
        chunks, current_chunk = [], ""
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk: continue
            if len(current_chunk) + len(chunk) <= max_length:
                current_chunk += " " + chunk
            else:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = chunk
        if current_chunk: chunks.append(current_chunk.strip())
        return [c for c in chunks if c]

    @web_app.post("/")
    async def process_audio(
        target_text: str = Form(...), 
        ref_audio: UploadFile = File(...)
    ):
        try:
            ref_bytes = await ref_audio.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                tmp_ref.write(ref_bytes)
                ref_path = tmp_ref.name

            text_chunks = split_text_into_chunks(target_text)
            generated_audio_list = []
            
            # 🚀 အသက်ရှူချိန် (Silence Pause) 0.4 စက္ကန့် ဖန်တီးခြင်း
            target_sample_rate = model.tts_model.sample_rate
            silence_chunk = np.zeros(int(target_sample_rate * 0.4), dtype=np.float32)
            
            for chunk in text_chunks:
                wav = model.generate(
                    text=chunk,
                    reference_wav_path=ref_path,
                    cfg_value=2.0,
                    inference_timesteps=10
                )
                
                # 🚀 အသံအတိုးအကျယ် (Volume) ကို အပိုင်းတိုင်းတွင် ပုံမှန်ဖြစ်အောင် ညှိပေးခြင်း (Normalization)
                max_val = np.max(np.abs(wav))
                if max_val > 0:
                    wav = (wav / max_val) * 0.9  # အသံကို 90% Level တွင် အမြဲတမ်း တစ်ပြေးညီ ထိန်းထားမည်
                    
                generated_audio_list.append(wav)
                generated_audio_list.append(silence_chunk) # စာကြောင်းတစ်ကြောင်းအပြီးတိုင်း အသက်ရှူချိန်ထည့်မည်

            final_audio = np.concatenate(generated_audio_list)
            
            # 🚀 တစ်ဖိုင်လုံးကို ဆက်ပြီးသားအချိန်တွင် ထပ်မံငြိမ်သွားစေရန် ဒုတိယအကြိမ် ညှိပေးခြင်း
            final_max = np.max(np.abs(final_audio))
            if final_max > 0:
                final_audio = (final_audio / final_max) * 0.95

            final_out_path = tempfile.mktemp(suffix=".wav")
            sf.write(final_out_path, final_audio, target_sample_rate)
            
            with open(final_out_path, 'rb') as f:
                final_bytes = f.read()
                
            os.unlink(ref_path)
            os.unlink(final_out_path)

            return Response(content=final_bytes, media_type="audio/wav")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"VoxCPM2 GPU Error: {str(e)}")
            
    return web_app
