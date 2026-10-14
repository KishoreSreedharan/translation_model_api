from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import translator_gemini_model

app = FastAPI(title = "Translation API")

class TranslateRequest(BaseModel):
    text:str

@app.post("/translate")
def trnaslating(req : TranslateRequest):
    try:
        return{"translation": translator_gemini_model.trnaslating(req.text)}
    except  ValueError  as e :
        raise HTTPException(status_code = 400, detail = str(e))

