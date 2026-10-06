import modal

# Modal ဆာဗာမှာ လိုအပ်တဲ့ စနစ်တွေ (ffmpeg အပါအဝင်) သွင်းခြင်း
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]", 
        "python-multipart",
        "gradio_client",
        "pydub"
    )
)

app = modal.App("voice-clone-api")

# အချိန်ကြာနိုင်တဲ့အတွက် timeout ကို ၆၀၀ စက္ကန့် (၁၀ မိနစ်) အထိ ပေးထားပါတယ်
@app.function(image=image, timeout=600)
@modal.asgi_app()
def generate_audio():
    # GitHub က မမြင်အောင် import တွေကို အတွင်းထဲ ရွှေ့ထားလိုက်ပါတယ်
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import Response
    import tempfile
    import os
    
    web_app = FastAPI()
    
    @web_app.post("/")
    async def process_audio(base_audio: UploadFile = File(...), ref_audio: UploadFile = File(...)):
        try:
            from gradio_client import Client, handle_file
            from pydub import AudioSegment
            
            # ပို့လိုက်တဲ့ အသံဖိုင်တွေကို ဖတ်မယ်
            base_bytes = await base_audio.read()
            ref_bytes = await ref_audio.read()
            
            # ယာယီဖိုင်လေးတွေ ဆောက်ပြီး မှတ်ထားမယ်
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_ref:
                tmp_ref.write(ref_bytes)
                ref_path = tmp_ref.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_base:
                tmp_base.write(base_bytes)
                base_path = tmp_base.name
                
            # အသံကို ၁၅ စက္ကန့်စီ ပိုင်းမယ်
            audio = AudioSegment.from_file(base_path)
            chunk_length_ms = 15000
            chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
            
            client = Client("hugging-apps/x-vc-voice-conversion")
            combined_audio = AudioSegment.empty()
            
            # အပိုင်းတစ်ပိုင်းချင်းစီကို အသံပွားမယ်
            for idx, chunk in enumerate(chunks):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_chunk:
                    chunk.export(tmp_chunk.name, format="wav")
                    chunk_path = tmp_chunk.name
                
                try:
                    result = client.predict(
                        source_audio=handle_file(chunk_path),
                        reference_audio=handle_file(ref_path),
                        mode="Offline (highest quality)",
                        chunk_ms=2400,
                        current_ms=120,
                        future_ms=100,
                        smooth_ms=20,
                        api_name="/convert"
                    )
                    converted_chunk = AudioSegment.from_file(result[0])
                    combined_audio += converted_chunk
                except Exception as chunk_err:
                    raise HTTPException(status_code=500, detail=f"Hugging Face က အပိုင်း {idx+1} ကို လက်မခံပါ: {str(chunk_err)}")
                finally:
                    os.unlink(chunk_path)
                    
            # အသံတွေပြန်ဆက်ပြီး Website ဆီ ပြန်ပို့မယ်
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_final:
                combined_audio.export(tmp_final.name, format="wav")
                with open(tmp_final.name, 'rb') as f:
                    final_bytes = f.read()
                    
            os.unlink(base_path)
            os.unlink(ref_path)
            os.unlink(tmp_final.name)
            
            return Response(content=final_bytes, media_type="audio/wav")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    return web_app
