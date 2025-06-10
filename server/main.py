from fastapi import FastAPI
from team_ops import Model
from pydantic import BaseModel

app = FastAPI()
model = Model()
model.load_model()


class Request(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"status": "Server is running"}


@app.post("/api/predict")
def predict_response(request: Request):
    """
    Experimental
    """
    try:
        prompt = request.prompt
        if prompt is None or prompt == "":
            raise Exception("Prompt cannot be empty")
        return {"status": "success", "message": model.predict(prompt)}
    except Exception as e:
        return {"status": "error", "message": e}
