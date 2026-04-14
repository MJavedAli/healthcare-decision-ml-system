from pydantic import BaseModel, Field


class AppointmentFeatures(BaseModel):
    age: int = Field(..., ge=0, le=120)
    prior_no_shows: int = Field(..., ge=0)
    lead_time_hours: float = Field(..., ge=0)
    num_previous_appointments: int = Field(..., ge=0)
    is_weekend: bool
    distance_km: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    no_show_probability: float
    predicted_label: int
    model_version: str
