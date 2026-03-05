from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from PIL import Image
import io

MODEL_PATH = Path("artifacts/model.joblib")

CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


class PredictRequest(BaseModel):
    pixels: Optional[List[float]] = Field(
        default=None,
        description="Length 784 array"
    )
    fill: Optional[float] = Field(
        default=None,
        description="Fill all pixels with one value"
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Generate deterministic random pixels"
    )


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    proba: List[float]


app = FastAPI(title="Fashion-MNIST Classic ML API")
_model = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training or dvc pull artifacts."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def _predict_array(x: np.ndarray):

    model = _load_model()

    if x.max() > 1.5:
        x = x / 255.0

    X = x.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        class_id = int(np.argmax(proba))
    else:
        class_id = int(model.predict(X)[0])
        proba = np.zeros(10)
        proba[class_id] = 1.0

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, str(class_id)),
        "proba": [float(p) for p in proba],
    }


@app.get("/health")
def health():
    ok = MODEL_PATH.exists()
    return {"status": "ok", "model_present": ok}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    if req.pixels is not None:

        if len(req.pixels) != 784:
            raise HTTPException(status_code=400, detail="pixels must have length 784")

        x = np.array(req.pixels, dtype=np.float32)

    elif req.fill is not None:

        x = np.full((784,), float(req.fill), dtype=np.float32)

    elif req.random_seed is not None:

        rng = np.random.default_rng(int(req.random_seed))
        x = rng.random(784)

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide pixels OR fill OR random_seed"
        )

    return _predict_array(x)


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    image = image.resize((28, 28))

    arr = np.array(image).astype(np.float32)
    arr = arr.flatten()

    return _predict_array(arr)


@app.get("/predict/random", response_model=PredictResponse)
def predict_random():

    x = np.random.random(784)

    return _predict_array(x)