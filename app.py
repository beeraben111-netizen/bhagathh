import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "romance": "Sita Ramam, Geetha Govindam, 96",
        "thriller": "Drishyam, Ratsasan, Andhadhun",
        "horror": "Tumbbad, Stree, Bhool Bhulaiyaa"
    }

    return movies.get(
        genre.lower().strip(),
        "No Indian movies found for that genre."
    )


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return temp_c * 1.8 + 32


@tool
def get_weather(city: str) -> str:
    """Get the current weather for an Indian city."""

    try:
        # ----------------------------------------------------
        # Geocoding
        # ----------------------------------------------------
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 10,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find weather data for city: {city}"

        # Prefer an Indian location if available
        location = None

        for result in geo_data["results"]:
            if result.get("country_code") == "IN":
                location = result
                break

        if location is None:
            location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # Weather API
        # ----------------------------------------------------
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data.get("current")

        if not current:
            return f"Could not retrieve current weather for {city}"

        result = {
            "resolved_city": location.get("name", city),
            "country": location.get("country", "India"),
            "temperature_celsius": current.get("temperature_2m"),
            "weather_code": current.get("weather_code")
        }

        return json.dumps(result)

    except requests.RequestException as e:
        return f"Weather service error: {str(e)}"

    except Exception as e:
        return f"Could not get weather data: {str(e)}"


# List of tools
tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# 2. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Add GEMINI_API_KEY in Render Environment Variables."
    )


# ============================================================
# 3. INITIALIZE GOOGLE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 4. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a specialized AI agent restricted ONLY to "
        "Indian weather and Indian cinema.\n\n"

        "You can:\n"
        "1. Answer questions about weather in Indian cities.\n"
        "2. Search/recommend Indian movies by genre.\n"
        "3. Convert Celsius temperatures to Fahrenheit when "
        "related to Indian weather.\n\n"

        "For weather questions, use the get_weather tool.\n"
        "For Indian movie questions, use the search_movies tool.\n"
        "For Celsius to Fahrenheit conversion, use the "
        "change_to_f tool.\n\n"

        "If the user asks about anything outside Indian weather "
        "and Indian cinema, you must respond exactly with:\n\n"

        "'I am not authorized to answer questions outside of "
        "Indian weather and cinema.'"
    )
)


# ============================================================
# 5. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the Indian Weather and Cinema Agent"
    )


# ============================================================
# 6. FORMAT INPUT FOR LANGCHAIN AGENT
# ============================================================

def format_for_agent(x):
    """
    Convert LangServe input into the message format
    expected by the LangChain agent.
    """

    if isinstance(x, dict):
        user_input = x.get("input", "")
    else:
        user_input = x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# 7. EXTRACT FINAL RESPONSE
# ============================================================

def extract_text_response(agent_output):
    """
    Extract the final assistant response from the
    LangChain agent output.
    """

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    # Handle nested agent output
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict):
                if "messages" in value:
                    messages = value["messages"]
                    break

    if messages:
        last_message = messages[-1]

        # LangChain message object
        content = getattr(last_message, "content", None)

        if content is not None:
            # Sometimes content can be a list
            if isinstance(content, list):
                text_parts = []

                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            text_parts.append(str(item["text"]))
                    else:
                        text_parts.append(str(item))

                return "".join(text_parts)

            return str(content)

        return str(last_message)

    return str(agent_output)


# ============================================================
# 8. CREATE LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 9. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    description=(
        "An AI agent specialized in Indian weather "
        "and Indian cinema."
    ),
    version="1.0.0"
)


# ============================================================
# 10. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 11. HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Indian Weather and Cinema Agent is running.",
        "endpoint": "/agent"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# 12. RUN SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
