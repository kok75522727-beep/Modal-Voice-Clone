import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App("voice-clone-api")

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def check_status():
    return {"status": "Success", "message": "API is working perfectly!"}
