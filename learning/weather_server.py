import os
import requests
from dotenv import load_dotenv
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

mcp = FastMCP("WeatherAssistant")

@mcp.prompt()
def compare_weather_prompt(location_a: str, location_b: str) -> str:
    return f"""
    You are acting as a helpful weather analyst. Your goal is to provide a clear and easy-to-read comparison of the weather in two different locations for a user.

    The user wants to compare the weather between "{location_a}" and "{location_b}".

    To accomplish this, follow these steps:
    1.  First, gather the necessary weather data for both "{location_a}" and "{location_b}".
    2.  Once you have the weather data for both locations, DO NOT simply list the raw results.
    3.  Instead, synthesize the information into a concise summary. Your final response should highlight the key differences, focusing on temperature, the general conditions (e.g., 'sunny' vs 'rainy'), and wind speed.
    4.  Present the comparison in a structured format, like a markdown table or a clear bulleted list, to make it easy for the user to understand at a glance.
    """

@mcp.resource("file://delivery_log")
def delivery_log() -> list[str]:
    try:
        log_file = Path("delivery_log.txt")
        if not log_file.exists():
            return ["Error: File not found on the server"]

        return log_file.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        return [f"Error reading log file: {e}"]

@mcp.tool()
def get_weather(location: str) -> dict:

    if not OPENWEATHERMAP_API_KEY:
        return {"error": "OpenWeatherMap API key is not set."}

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        return {
            "location": location,
            "weather": weather_description,
            "temperature": temperature,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind_speed": wind_speed
        }
    
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return {"error": f"Couldn't find weather for '{location}'. Please check the name of the location"}
        elif response.status_code == 401:
            return {"error": "Unauthorized access. Please check your API key."}
        else:
            return {"error": f"An error occurred: {http_err}"}
    except requests.RequestException as e:
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Missing data in response: {e}"}
    except (ValueError, TypeError) as e:
        return {"error": f"Data processing error: {e}"}
    
if __name__ == "__main__":
    print("Starting Weather MCP Server...")
    mcp.run(transport="stdio")
