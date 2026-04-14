import pytest
from pydantic import ValidationError

from src.common.schema import AppointmentFeatures


def test_schema_accepts_valid_input():
    obj = AppointmentFeatures(
        age=34,
        prior_no_shows=2,
        lead_time_hours=24.0,
        num_previous_appointments=7,
        is_weekend=False,
        distance_km=10.5,
    )
    assert obj.age == 34


def test_schema_rejects_negative_age():
    with pytest.raises(ValidationError):
        AppointmentFeatures(
            age=-1,
            prior_no_shows=2,
            lead_time_hours=24.0,
            num_previous_appointments=7,
            is_weekend=False,
            distance_km=10.5,
        )
