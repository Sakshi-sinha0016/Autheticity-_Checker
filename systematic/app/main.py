# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.predictor import predict_profile

app = FastAPI()

# Pydantic models
class ProfileData(BaseModel):
    headline: str
    bio: str

class ProfileRequest(BaseModel):
    user_id: str
    profile_data: ProfileData

# Endpoints
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": "1.0.0"}

@app.post("/check-profile")
def check_profile(request: ProfileRequest):
    try:
        result = predict_profile(
            request.profile_data.headline,
            request.profile_data.bio
        )
        return {
            "user_id": request.user_id,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
