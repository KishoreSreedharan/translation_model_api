from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from indic_trans2_model import translate

app = FastAPI(title="Translation API")

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    try:
        return {"translation": translate(req.text)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))