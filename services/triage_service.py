
from pydantic import BaseModel, Field
from typing import List, Literal

class TriageService(BaseModel):
    health_overview: str = Field(description="High-level summary of the pet's overall condition based on reported symptoms and conversation.")

    symptoms_identified: List[str] = Field(description="List of distinct symptoms explicitly mentioned or clearly inferred from the user's description.")

    symptom_analysis: str = Field(description=("Non-diagnostic reasoning of how the symptoms may be related." "Must avoid definitive medical diagnoses. Use cautious language such as 'may indicate'."))
    
    risk_level: Literal["low", "moderate", "high", "emergency"] = Field(description="Urgency category used strictly for escalation logic.")

    triage_category: Literal["red", "orange", "yellow", "green"] = Field(
        description=(
            "Triage classification based on severity guidelines: "
            "red = true emergency, like seizures,"
            "orange = urgent, like severe vomiting, "
            "yellow = semi-urgent, like allergies,  "
            "green = non-urgent for example for hair loss.") )
    
    recommendations: str = Field(description="Clear, conversational guidance written directly to the pet owner speak as you are from PawsPalConnect assistant in a warm, calm tone suitable for text-to-speech. Escalate immediately when risk_level is emergency. Include disclaimer about not replacing professional vet care. If no health concern, respond warmly to the conversation.")
    
    safety_flags: List[str] = Field(description="Independent safety triggers detected from symptoms (e.g., 'bleeding_present').")
    
    key_details: List[str] = Field(description="Key contextual items (age, names, greetings etc). Add NEW details only — do not repeat what's already in previous_context.")

