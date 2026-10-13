from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import translator_key_model

app = FastAPI(title = "Translation API")

class TranslateRequest(BaseModel):
    text:str

@app.post("/translate")
def translate(req : TranslateRequest):
    try:
        return{"translation": translator_key_model.translate(req.text)}
    except  ValueError  as e :
        raise HTTPException(status_code = 400, detail = str(e))

