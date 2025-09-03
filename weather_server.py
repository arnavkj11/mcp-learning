import os
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

mcp = FastMCP("WeatherAssistant")

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
    mcp.run(transport="stdio")
