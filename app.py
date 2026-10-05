import modal

app = modal.App("voice-clone-api")

@app.function()
@modal.fastapi_endpoint(method="GET")
def check_status():
    return {"status": "Success", "message": "API is working perfectly!"}
