
from pydantic import BaseModel, Field
from typing import List, Literal

class TriageOutput(BaseModel):
    health_overview: str = Field(
        description="High-level summary of the pet's overall condition based on reported symptoms and conversation."
    )

    symptoms_identified: List[str] = Field(
        description="List of distinct symptoms explicitly mentioned or clearly inferred from the user's description."
    )

    symptom_analysis: str = Field(
        description=(
            "Non-diagnostic reasoning of how the symptoms may be related. "
            "Must avoid definitive medical diagnoses. Use cautious language such as "
            "'may indicate'."
        )
    )

    risk_level: Literal["low", "moderate", "high", "emergency"] = Field(
        description="Urgency category used strictly for escalation logic."
    )

    triage_category: Literal["red", "orange", "yellow", "green"] = Field(
        description=(
            "Triage classification based on severity guidelines: "
            "red = true emergency, like seizures,"
            "orange = urgent, like severe vomiting, "
            "yellow = semi-urgent, like allergies,  "
            "green = non-urgent for example for hair loss."
        )
    )
    recommendations: str = Field(
        description="Clear, safety-aware guidance for the pet owner. Must escalate immediately when risk_level is emergency, red."
    )

    safety_flags: List[str] = Field(
        description="Independent safety triggers detected from symptoms (e.g., 'bleeding_present')."
    )