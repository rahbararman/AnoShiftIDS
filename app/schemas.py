from typing import List

from pydantic import BaseModel, validator


class TrafficData(BaseModel):
    record: List[str]

    @validator("record")
    def record_must_have_14_features(cls, value):
        if not len(value) == 14:
            raise ValueError("Traffic record must have 14 features.")
        return value


class PredictPayload(BaseModel):
    records: List[TrafficData]

    class Config:
        schema_extra = {
            "example": {
                "records": [
                    {
                        "record": [
                            "c041",
                            "other",
                            "c263",
                            "c363",
                            "0",
                            "0.0",
                            "0.0",
                            "0.41",
                            "0",
                            "0",
                            "0.0",
                            "0.0",
                            "0.0",
                            "SF",
                        ]
                    },
                    {
                        "record": [
                            "c041",
                            "other",
                            "c263",
                            "c363",
                            "0",
                            "0.0",
                            "0.0",
                            "0.45",
                            "0",
                            "0",
                            "0.0",
                            "0.0",
                            "0.0",
                            "SF",
                        ]
                    },
                ]
            }
        }
