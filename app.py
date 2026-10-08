import modal

def download_models():
    from transformers import AutoModel, AutoTokenizer
    # VoxCPM2 AI မော်ဒယ်အကြီးစားကို ကြိုတင် ဒေါင်းလုဒ်ဆွဲထားမည်
    model_id = "openbmb/VoxCPM2"
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    AutoModel.from_pretrained(model_id, trust_remote_code=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .run_commands(
        "pip install --upgrade pip",
        "pip install torch torchaudio transformers soundfile librosa fastapi[standard] python-multipart pydub numpy"
    )
    .run_function(download_models)
)

app = modal.App("oneteam-voice-clone-pro")

@app.function(image=image, gpu="T4", timeout=1500)
@modal.asgi_app()
def my_voice_clone_api():
    from fastapi import FastAPI, UploadFile, File, HTTPException, Form
    from fastapi.responses import Response
    import tempfile
    import os
    import torch
    import librosa
    import numpy as np
    import soundfile as sf
    from transformers import AutoModel, AutoTokenizer
    import re

    web_app = FastAPI()

    # Model ကို GPU ပေါ် တင်မည်
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openbmb/VoxCPM2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    # စာရှည်ပါက Memory မပြည့်အောင် အလိုအလျောက် ပိုင်းဖြတ်မည့်စနစ်
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
        target_text: str = Form(...), # 🚀 ယခုအခါ မြန်မာစာကို တိုက်ရိုက် လက်ခံမည်
        ref_audio: UploadFile = File(...)
    ):
        try:
            ref_bytes = await ref_audio.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                tmp_ref.write(ref_bytes)
                ref_path = tmp_ref.name

            text_chunks = split_text_into_chunks(target_text)
            generated_audio_list = []
            target_sample_rate = 24000
            
            # Reference Voice ဖတ်ခြင်း
            ref_audio_data, sr = librosa.load(ref_path, sr=16000)
            ref_audio_tensor = torch.tensor(ref_audio_data).unsqueeze(0).to(device)

            # AI မှ မြန်မာစာကို တိုက်ရိုက်ဖတ်၍ အသံပြောင်းခြင်း
            for chunk in text_chunks:
                inputs = tokenizer(chunk, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        prompt_audio=ref_audio_tensor,
                        prompt_sample_rate=16000,
                        output_sample_rate=target_sample_rate
                    )
                    if isinstance(output, torch.Tensor):
                        audio_chunk = output.squeeze().cpu().numpy()
                    else:
                        audio_chunk = output
                    generated_audio_list.append(audio_chunk)

            # အသံများ ပြန်ပေါင်းခြင်း
            final_audio = np.concatenate(generated_audio_list)
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
