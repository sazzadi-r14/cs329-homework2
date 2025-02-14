import os
import requests
from datetime import datetime
from openai import OpenAI
import json
from googleapiclient.discovery import build
from textblob import TextBlob
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote
import pytz

class APIManager:
    """A unified class to manage various API interactions"""
    
    def __init__(self, google_api_key: str, google_cx_id: str, alpha_vantage_key: str):
        """
        INPUT:
            google_api_key: str - Google API key for Custom Search
            google_cx_id: str - Google Custom Search Engine ID
            alpha_vantage_key: str - Alpha Vantage API key
            
        Initializes API keys and configurations
        """
        ################ CODE STARTS HERE ###############

        self.google_api_key = google_api_key
        self.google_cx_id = google_cx_id
        self.alpha_vantage_key = alpha_vantage_key

        ################ CODE ENDS HERE ###############

    def parse_query_params(self, query: str, function_name: str) -> Optional[Dict]:
        """
        INPUT:
            query: str - Natural language query from user
            function_name: str - Name of the function to parse parameters for
            
        OUTPUT:
            Optional[Dict] - Parameters needed for the specified function or None if parsing fails
        """
        ################ CODE STARTS HERE ###############

        pass

        ################ CODE ENDS HERE ###############

    def route_query(self, query: str) -> Dict:
        """
        INPUT:
            query: str - Natural language query to route
            
        OUTPUT:
            Dict containing:
                - results: Any - Results from the API call
                - api_used: str - Name of the API that was used
                - error: str (optional) - Error message if something went wrong
        """
        ################ CODE STARTS HERE ###############

        pass

        ################ CODE ENDS HERE ###############
        
    def _fetch_webpage_content(self, url: str) -> Optional[Dict]:
        """
        INPUT:
            url: str - URL to fetch
            
        OUTPUT:
            Optional[Dict] containing:
                - full_text: str - Full webpage text
                - main_content: str - Main content section
                - title: str - Page title
                - meta_description: str - Meta description
            Returns None if fetch fails
        """
        ################ CODE STARTS HERE ###############
        try:
            # Validate and parse URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                print(f"Invalid URL format: {url}")
                return None
            
            # Encode URL properly
            encoded_url = quote(url, safe=':/?=&')
            
            # Send GET request with a user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(encoded_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                print(f"Skipping non-HTML content: {content_type}")
                return None
            
            # Parse the webpage content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Extract title
            title = soup.title.string.strip() if soup.title else ''
            
            # Extract meta description
            meta_desc = ''
            meta_tags = [
                soup.find('meta', attrs={'name': 'description'}),
                soup.find('meta', attrs={'property': 'og:description'}),
                soup.find('meta', attrs={'name': 'twitter:description'})
            ]
            for meta_tag in meta_tags:
                if meta_tag and meta_tag.get('content'):
                    meta_desc = meta_tag.get('content', '').strip()
                    break
            
            # Get main content (try common content containers)
            main_content = ''
            content_tags = soup.find_all(['article', 'main', 'div'], 
                                       class_=['content', 'main', 'article', 'post', 'entry-content', 'page-content'])
            if content_tags:
                main_content = ' '.join(tag.get_text(strip=True, separator=' ') for tag in content_tags)
            else:
                # Fallback to body content if no main content containers found
                body = soup.find('body')
                if body:
                    # Remove common non-content areas from body
                    for elem in body.find_all(['header', 'footer', 'nav', 'sidebar']):
                        elem.decompose()
                    main_content = body.get_text(strip=True, separator=' ')
            
            # Get full text (excluding removed elements)
            full_text = soup.get_text(strip=True, separator=' ')
            
            # Clean up text (remove excessive whitespace)
            full_text = ' '.join(full_text.split())
            main_content = ' '.join(main_content.split())
            
            return {
                'full_text': full_text[:10000],  # Limit text length to avoid huge responses
                'main_content': main_content[:5000],  # Limit main content length
                'title': title,
                'meta_description': meta_desc
            }
            
        except Exception as e:
            print(f"Failed to fetch webpage content: {str(e)}")
            return None

        ################ CODE ENDS HERE ###############
    
    def google_search(self, search_term: str, num_results: int = 10) -> List[Dict]:
        """
        INPUT:
            search_term: str - The search query
            num_results: int - Number of results to return (default: 10)
            
        OUTPUT:
            List[Dict] - List of search results, each containing:
                - title: str
                - link: str
                - snippet: str
                - api_content: Dict with formatted and plain content
                - webpage_content: Dict (optional)
        """
        ################ CODE STARTS HERE ###############
        try:
            # Build the Custom Search API service
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            
            # Execute the search request
            result = service.cse().list(
                q=search_term,
                cx=self.google_cx_id,
                num=min(num_results, 10)  # API limits to 10 results per request
            ).execute()

            # Process results
            search_results = []
            if 'items' in result:
                for item in result['items']:
                    search_result = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'api_content': {
                            'formatted': item.get('htmlSnippet', ''),
                            'plain': item.get('snippet', ''),
                            'html_title': item.get('htmlTitle', ''),
                            'formatted_url': item.get('formattedUrl', ''),
                            'html_formatted_url': item.get('htmlFormattedUrl', ''),
                            'display_link': item.get('displayLink', '')
                        }
                    }
                    
                    # Add pagemap data if available
                    if 'pagemap' in item:
                        search_result['api_content']['pagemap'] = item['pagemap']
                    
                    # Attempt to fetch webpage content if available
                    try:
                        webpage_content = self._fetch_webpage_content(item.get('link', ''))
                        if webpage_content:
                            search_result['webpage_content'] = webpage_content['full_text']
                    except Exception as e:
                        search_result['webpage_content'] = None
                    
                    search_results.append(search_result)
                    
                    # Break if we have enough results
                    if len(search_results) >= num_results:
                        break
                    
            return search_results
            
        except Exception as e:
            # Return empty list if search fails
            print(f"Google search failed: {str(e)}")
            return []

        ################ CODE ENDS HERE ###############
    
    def get_stock_data(self, symbol: str, date: Optional[str] = None) -> Dict:
        """
        INPUT:
            symbol: str - Stock symbol (e.g., 'AAPL')
            date: Optional[str] - Date in format 'YYYY-MM-DD'
            
        OUTPUT:
            Dict containing either:
                Current data:
                    - symbol: str
                    - price: float
                    - change: float
                    - change_percent: str
                Historical data:
                    - date: str
                    - open: float
                    - high: float
                    - low: float
                    - close: float
                    - volume: int
        """
        ################ CODE STARTS HERE ###############
        try:
            base_url = "https://www.alphavantage.co/query"
            
            if date:
                # Get historical data
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'apikey': self.alpha_vantage_key,
                    'outputsize': 'full'  # Get full data to ensure we have the requested date
                }
                
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Check for error messages
                if "Error Message" in data:
                    print(f"API Error: {data['Error Message']}")
                    return {}
                    
                # Get daily time series data
                time_series = data.get('Time Series (Daily)', {})
                
                # Look for the requested date
                if date in time_series:
                    daily_data = time_series[date]
                    return {
                        'date': date,
                        'open': float(daily_data['1. open']),
                        'high': float(daily_data['2. high']),
                        'low': float(daily_data['3. low']),
                        'close': float(daily_data['4. close']),
                        'volume': int(daily_data['5. volume'])
                    }
                else:
                    print(f"No data available for date: {date}")
                    return {}
                    
            else:
                # Get current quote data
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.alpha_vantage_key
                }
                
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Check for error messages
                if "Error Message" in data:
                    print(f"API Error: {data['Error Message']}")
                    return {}
                    
                quote = data.get('Global Quote', {})
                if quote:
                    return {
                        'symbol': quote.get('01. symbol', symbol),
                        'price': float(quote.get('05. price', 0)),
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': quote.get('10. change percent', '0%')
                    }
                else:
                    print(f"No current data available for symbol: {symbol}")
                    return {}
                    
        except Exception as e:
            print(f"Failed to fetch stock data: {str(e)}")
            return {}

        ################ CODE ENDS HERE ###############
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict:
        """
        INPUT:
            text: str - Text to analyze
            
        OUTPUT:
            Dict containing:
                - sentiment: str - "positive", "negative", or "neutral"
                - polarity: float - Sentiment polarity score
                - subjectivity: float - Subjectivity score
        """
        ################ CODE STARTS HERE ###############

        class SentimentSchema(BaseModel):
            sentiment: str
            polarity: float 
            subjectivity: float

        openai = OpenAI()

        prompt = f"""Analyze the sentiment of the following text and provide a structured response.

Text to analyze: {text}

Guidelines for analysis:
- Determine if the overall sentiment is positive, negative, or neutral
- Calculate a polarity score from -1.0 (most negative) to 1.0 (most positive)
- Calculate a subjectivity score from 0.0 (most objective) to 1.0 (most subjective)

Required output format:
{{
    "sentiment": "positive|negative|neutral",
    "polarity": <float between -1.0 and 1.0>,
    "subjectivity": <float between 0.0 and 1.0>
}}"""

        completion = openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Provide analysis results in the exact JSON format specified."},
                {"role": "user", "content": prompt}
            ],
            response_format=SentimentSchema
        )

        result = completion.choices[0].message.content
        return json.loads(result)

        ################ CODE ENDS HERE ###############
    
    @staticmethod
    def get_weather(location: str, date: str, hour: str = "12") -> Dict:
        """
        INPUT:
            location: str - Location string (e.g., "Palo Alto, CA")
            date: str - Date in YYYY-MM-DD format
            hour: str - Hour in 24-hour format (default: "12")
            
        OUTPUT:
            Dict containing:
                - temperature: str
                - weather_description: str
                - humidity: str
                - wind_speed: str, any wind speed value is acceptable
        """
        ################ CODE STARTS HERE ###############
        print(f"\nGetting weather for location: {location}, date: {date}, hour: {hour}")
        
        lat, lon = APIManager._get_coordinates(location)
        print(f"Coordinates retrieved - lat: {lat}, lon: {lon}")
        
        if lat is None or lon is None:
            # If coordinates couldn't be retrieved, return an error indication (or raise an exception)
            return {"error": f"Could not find location: {location}"}
        
        url = "https://api.openweathermap.org/data/3.0/onecall"
        params = {
            "lat": lat,
            "lon": lon,
            "exclude": "minutely,hourly,daily,alerts",  # exclude all but current
            "appid": os.getenv("OPENWEATHER_API_KEY"),
            "units": "metric"  # use Celsius; use "imperial" for Fahrenheit or omit for Kelvin
        }
        try:
            print("Making API request to OpenWeather...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            print("API request successful")
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {"error": "Failed to retrieve weather data"}
        
        data = response.json()
        print(f"Response received with status code: {response.status_code}")
        
        current = data.get("current", {})
        # Extract fields from the current weather data
        temperature = current.get("temp")
        humidity = current.get("humidity")
        wind_speed = current.get("wind_speed")
        # 'weather' is a list of weather conditions; we take the first element for current weather
        weather_list = current.get("weather", [])
        if weather_list:
            main_condition = weather_list[0].get("main")        # e.g., "Rain"
            description_text = weather_list[0].get("description")  # e.g., "light rain"
            condition_code = weather_list[0].get("id")         # e.g., 500
        else:
            main_condition = description_text = ""
            condition_code = None

        # Map the condition code to a human-readable description (if code exists)
        weather_description = APIManager._get_weather_description(condition_code) if condition_code is not None else ""

        # Prepare the result dictionary
        result = {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "conditions": main_condition,
            "weather_description": weather_description or description_text
        }
        print(f"Weather data retrieved successfully: {result}")
        return result

        ################ CODE ENDS HERE ###############
    
    @staticmethod
    def _get_coordinates(location: str) -> Optional[tuple]:
        """
        INPUT:
            location: str - Location name to geocode
            
        OUTPUT:
            Optional[tuple] - (latitude: float, longitude: float) or None if not found
        """
        ################ CODE STARTS HERE ###############
        print(f"\nGetting coordinates for location: {location}")
        url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {"q": location, "limit": 1, "appid": "873b7e245edcc01cb2da60b2a3562e9a"}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises an HTTPError if the status is 4xx/5xx
            print(response)
            print(response.json())
            print("Geocoding API request successful")
        except requests.RequestException as e:
            print(f"Error fetching coordinates: {e}")
            return None, None

        data = response.json()
        if not data:
            # No results found for the location
            print(f"No coordinates found for '{location}'.")
            return None, None

        # Extract latitude and longitude from the first result
        latitude = data[0].get("lat")
        longitude = data[0].get("lon")
        print(f"Coordinates found - lat: {latitude}, lon: {longitude}")
        return latitude, longitude


        ################ CODE ENDS HERE ###############
    
    @staticmethod
    def _get_weather_description(code: int) -> str:
        """
        INPUT:
            code: int - Weather condition code
            
        OUTPUT:
            str - Human-readable weather description
        """
        ################ CODE STARTS HERE ###############
        if code is None:
            return ""
        if 200 <= code < 300:
            return "Thunderstorm"        # Group 2xx: Thunderstorm&#8203;:contentReference[oaicite:8]{index=8}
        elif 300 <= code < 400:
            return "Drizzle"             # Group 3xx: Drizzle
        elif 500 <= code < 600:
            return "Rain"                # Group 5xx: Rain&#8203;:contentReference[oaicite:9]{index=9}
        elif 600 <= code < 700:
            return "Snow"                # Group 6xx: Snow
        elif 700 <= code < 800:
            return "Atmospheric Haze"    # Group 7xx: Mist, fog, smoke, etc.
        elif code == 800:
            return "Clear sky"           # 800: Clear weather&#8203;:contentReference[oaicite:10]{index=10}
        elif 801 <= code < 900:
            return "Cloudy"              # 80x: Clouds (few, scattered, broken, overcast)
        else:
            return "Unknown"

        ################ CODE ENDS HERE ###############