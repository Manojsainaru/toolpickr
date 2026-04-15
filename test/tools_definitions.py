import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolpickr.core.tool import ToolDefinition

tools = [
    # 🌍 GENERAL / SEARCH
    ToolDefinition(
        name="web_search",
        description="Search the web for information.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to look up on the web."},
            },
            "required": ["query"]
        }
    ),
    ToolDefinition(
        name="news_search",
        description="Search latest news articles.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The topic or keywords to search in recent news."},
            },
            "required": ["query"]
        }
    ),
    ToolDefinition(
        name="image_search",
        description="Search for images.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search terms to find relevant images."},
            },
            "required": ["query"]
        }
    ),
    ToolDefinition(
        name="video_search",
        description="Search for videos.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search terms to find relevant videos."},
            },
            "required": ["query"]
        }
    ),

    # 🌦 WEATHER
    ToolDefinition(
        name="get_weather",
        description="Get current weather.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state/country to get the weather for."},
            },
            "required": ["location"]
        }
    ),
    ToolDefinition(
        name="get_weather_forecast",
        description="Get weather forecast.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state/country to get the forecast for."},
                "days": {"type": "number", "description": "The number of days into the future for the forecast."},
            },
            "required": ["location", "days"]
        }
    ),

    # 💰 FINANCE
    ToolDefinition(
        name="get_asset_data",
        description="Fetch financial data for an asset.",
        parameters={
            "type": "object",
            "properties": {
                "asset_type": {"type": "string", "description": "The type of asset, e.g., 'stock', 'crypto', 'bond'."},
                "symbol": {"type": "string", "description": "The ticker symbol of the asset (e.g., AAPL)."},
                "fields": {"type": "list", "description": "List of data fields to fetch (e.g., ['price', 'volume'])."},
            },
            "required": ["asset_type", "symbol", "fields"]
        }
    ),
    ToolDefinition(
        name="get_asset_news",
        description="Get news for an asset.",
        parameters={
            "type": "object",
            "properties": {
                "asset_type": {"type": "string", "description": "The type of asset."},
                "symbol": {"type": "string", "description": "The ticker symbol of the asset."},
            },
            "required": ["asset_type", "symbol"]
        }
    ),
    ToolDefinition(
        name="convert_currency",
        description="Convert currency.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "The amount to convert."},
                "from": {"type": "string", "description": "The base currency code (e.g., USD)."},
                "to": {"type": "string", "description": "The target currency code (e.g., EUR)."},
            },
            "required": ["amount", "from", "to"]
        }
    ),
    ToolDefinition(
        name="calculate_investment_return",
        description="Calculate investment return.",
        parameters={
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "The initial investment amount."},
                "rate": {"type": "number", "description": "The annual interest rate (as a decimal)."},
                "time": {"type": "number", "description": "The time period of the investment in years."},
            },
            "required": ["principal", "rate", "time"]
        }
    ),

    # 🛒 E-COMMERCE
    ToolDefinition(
        name="search_products",
        description="Search for products.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The name or category of the product to search for."},
            },
            "required": ["query"]
        }
    ),
    ToolDefinition(
        name="get_product_details",
        description="Get product details.",
        parameters={
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The unique identifier of the product."},
            },
            "required": ["product_id"]
        }
    ),
    ToolDefinition(
        name="compare_products",
        description="Compare products.",
        parameters={
            "type": "object",
            "properties": {
                "product_ids": {"type": "list", "description": "A list of product IDs to compare against each other."},
            },
            "required": ["product_ids"]
        }
    ),
    ToolDefinition(
        name="track_order",
        description="Track an order.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The tracking or order ID of the purchase."},
            },
            "required": ["order_id"]
        }
    ),

    # 📍 MAPS & TRAVEL
    ToolDefinition(
        name="find_places",
        description="Find nearby places.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The type of place to find, like 'coffee shop' or 'gas station'."},
                "location": {"type": "string", "description": "The center location to search around."},
            },
            "required": ["query", "location"]
        }
    ),
    ToolDefinition(
        name="get_directions",
        description="Get directions.",
        parameters={
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "The starting address or location."},
                "to": {"type": "string", "description": "The destination address or location."},
            },
            "required": ["from", "to"]
        }
    ),
    ToolDefinition(
        name="get_distance",
        description="Calculate distance.",
        parameters={
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Starting address or location."},
                "to": {"type": "string", "description": "Ending address or location."},
            },
            "required": ["from", "to"]
        }
    ),
    ToolDefinition(
        name="book_flight",
        description="Book a flight.",
        parameters={
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Departure airport code or city."},
                "to": {"type": "string", "description": "Arrival airport code or city."},
                "date": {"type": "string", "description": "Date of flight, in YYYY-MM-DD format."},
            },
            "required": ["from", "to", "date"]
        }
    ),
    ToolDefinition(
        name="search_hotels",
        description="Search hotels.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city or neighborhood to look for a hotel."},
                "checkin": {"type": "string", "description": "Check-in date (YYYY-MM-DD)."},
                "checkout": {"type": "string", "description": "Check-out date (YYYY-MM-DD)."},
            },
            "required": ["location", "checkin", "checkout"]
        }
    ),

    # 🍔 FOOD
    ToolDefinition(
        name="find_restaurants",
        description="Find restaurants.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Location to search for restaurants."},
                "cuisine": {"type": "string", "description": "Type of food or cuisine desired (e.g., Italian, Mexican)."},
            },
            "required": ["location", "cuisine"]
        }
    ),
    ToolDefinition(
        name="get_menu",
        description="Get restaurant menu.",
        parameters={
            "type": "object",
            "properties": {
                "restaurant_name": {"type": "string", "description": "The exact name of the restaurant."},
            },
            "required": ["restaurant_name"]
        }
    ),
    ToolDefinition(
        name="order_food",
        description="Order food.",
        parameters={
            "type": "object",
            "properties": {
                "restaurant": {"type": "string", "description": "Name of the restaurant to order from."},
                "items": {"type": "list", "description": "List of food items and quantities to order."},
            },
            "required": ["restaurant", "items"]
        }
    ),

    # 📅 PRODUCTIVITY
    ToolDefinition(
        name="create_calendar_event",
        description="Create calendar event.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title or subject of the meeting."},
                "date": {"type": "string", "description": "The event date in YYYY-MM-DD format."},
                "time": {"type": "string", "description": "The event time in HH:MM format."},
            },
            "required": ["title", "date", "time"]
        }
    ),
    ToolDefinition(
        name="get_calendar_events",
        description="Get calendar events.",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "The date to retrieve your schedule for (YYYY-MM-DD)."},
            },
            "required": ["date"]
        }
    ),
    ToolDefinition(
        name="set_reminder",
        description="Set reminder.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Description of the task to be reminded about."},
                "time": {"type": "string", "description": "Date and/or time to trigger the reminder."},
            },
            "required": ["task", "time"]
        }
    ),
    ToolDefinition(
        name="send_email",
        description="Send email.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Email address of the recipient."},
                "subject": {"type": "string", "description": "Subject line of the email."},
                "body": {"type": "string", "description": "Main text content of the email."},
            },
            "required": ["to", "subject", "body"]
        }
    ),

    # 📂 FILE SYSTEM
    ToolDefinition(
        name="read_file",
        description="Read file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path to read."},
            },
            "required": ["path"]
        }
    ),
    ToolDefinition(
        name="write_file",
        description="Write file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to be written to."},
                "content": {"type": "string", "description": "The string content to write inside the file."},
            },
            "required": ["path", "content"]
        }
    ),
    ToolDefinition(
        name="delete_file",
        description="Delete file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to delete."},
            },
            "required": ["path"]
        }
    ),
    ToolDefinition(
        name="list_files",
        description="List files in directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the directory to list the files of."},
            },
            "required": ["path"]
        }
    ),

    # 🧮 UTILITIES
    ToolDefinition(
        name="calculator",
        description="Evaluate math expression.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A mathematical expression to solve (e.g., '2 + 2 * 4')."},
            },
            "required": ["expression"]
        }
    ),
    ToolDefinition(
        name="unit_convert",
        description="Convert units.",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numerical value to convert."},
                "from": {"type": "string", "description": "The base unit of measurement (e.g., 'lbs')."},
                "to": {"type": "string", "description": "The target unit to convert into (e.g., 'kg')."},
            },
            "required": ["value", "from", "to"]
        }
    ),
    ToolDefinition(
        name="get_time",
        description="Get current time.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city or timezone to get the current time of."},
            },
            "required": ["location"]
        }
    ),

    # 🧠 KNOWLEDGE / AI
    ToolDefinition(
        name="ask_knowledge_base",
        description="Query internal knowledge base.",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask the company knowledge base."},
            },
            "required": ["question"]
        }
    ),
    ToolDefinition(
        name="summarize_text",
        description="Summarize text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "A long text string to be summarized into key points."},
            },
            "required": ["text"]
        }
    ),
    ToolDefinition(
        name="translate_text",
        description="Translate text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The sequence of characters or sentences to translate."},
                "target_language": {"type": "string", "description": "The language code or name to translate into."},
            },
            "required": ["text", "target_language"]
        }
    ),
    ToolDefinition(
        name="extract_entities",
        description="Extract entities from text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to process for extracting named entities like people and organizations."},
            },
            "required": ["text"]
        }
    ),

    # 🏥 HEALTH
    ToolDefinition(
        name="search_symptoms",
        description="Search symptoms.",
        parameters={
            "type": "object",
            "properties": {
                "symptoms": {"type": "string", "description": "Comma-separated list of medical symptoms."},
            },
            "required": ["symptoms"]
        }
    ),
    ToolDefinition(
        name="find_doctors",
        description="Find doctors.",
        parameters={
            "type": "object",
            "properties": {
                "specialty": {"type": "string", "description": "The medical specialization, e.g., 'Cardiologist'."},
                "location": {"type": "string", "description": "Zip code or city to find a doctor in."},
            },
            "required": ["specialty", "location"]
        }
    ),
    ToolDefinition(
        name="book_appointment",
        description="Book doctor appointment.",
        parameters={
            "type": "object",
            "properties": {
                "doctor": {"type": "string", "description": "The name or ID of the specific doctor."},
                "date": {"type": "string", "description": "The requested date and time for the appointment."},
            },
            "required": ["doctor", "date"]
        }
    ),

    # 🎓 EDUCATION
    ToolDefinition(
        name="search_courses",
        description="Search online courses.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The subject or skill to find a course for."},
            },
            "required": ["query"]
        }
    ),
    ToolDefinition(
        name="get_course_details",
        description="Get course details.",
        parameters={
            "type": "object",
            "properties": {
                "course_id": {"type": "string", "description": "The specific ID of the course."},
            },
            "required": ["course_id"]
        }
    ),
    ToolDefinition(
        name="recommend_books",
        description="Recommend books.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The genre, author, or subject area to base book recommendations on."},
            },
            "required": ["topic"]
        }
    ),

    # 🎵 MEDIA
    ToolDefinition(
        name="play_music",
        description="Play music.",
        parameters={
            "type": "object",
            "properties": {
                "song": {"type": "string", "description": "The name of the song, artist, or playlist to play."},
            },
            "required": ["song"]
        }
    ),
    ToolDefinition(
        name="get_song_info",
        description="Get song info.",
        parameters={
            "type": "object",
            "properties": {
                "song": {"type": "string", "description": "The name of the song to search for lyrics or metadata."},
            },
            "required": ["song"]
        }
    ),
    ToolDefinition(
        name="recommend_movies",
        description="Recommend movies.",
        parameters={
            "type": "object",
            "properties": {
                "genre": {"type": "string", "description": "Movie genre to fetch recommendations for (e.g., 'Sci-Fi')."},
            },
            "required": ["genre"]
        }
    ),
    ToolDefinition(
        name="get_movie_details",
        description="Get movie details.",
        parameters={
            "type": "object",
            "properties": {
                "movie": {"type": "string", "description": "The exact title or ID of the movie."},
            },
            "required": ["movie"]
        }
    ),

    # 🚗 TRANSPORT
    ToolDefinition(
        name="book_ride",
        description="Book a ride.",
        parameters={
            "type": "object",
            "properties": {
                "pickup": {"type": "string", "description": "The address or location where the driver should pick you up."},
                "drop": {"type": "string", "description": "The destination address where you want to be dropped off."},
            },
            "required": ["pickup", "drop"]
        }
    ),
    ToolDefinition(
        name="get_ride_estimate",
        description="Get ride estimate.",
        parameters={
            "type": "object",
            "properties": {
                "pickup": {"type": "string", "description": "The starting location of the ride."},
                "drop": {"type": "string", "description": "The ending destination of the ride."},
            },
            "required": ["pickup", "drop"]
        }
    ),

    # 💼 JOBS
    ToolDefinition(
        name="search_jobs",
        description="Search jobs.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Job title, keywords, or company to search for."},
                "location": {"type": "string", "description": "City, state, or remote location to constrain the job search."},
            },
            "required": ["query", "location"]
        }
    ),
    ToolDefinition(
        name="get_job_details",
        description="Get job details.",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The unique identifier for the job listing."},
            },
            "required": ["job_id"]
        }
    ),

    # 🏠 REAL ESTATE
    ToolDefinition(
        name="search_properties",
        description="Search properties.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city or zip code to look for real estate in."},
                "budget": {"type": "number", "description": "The maximum price or budget for the property."},
            },
            "required": ["location", "budget"]
        }
    ),
    ToolDefinition(
        name="get_property_details",
        description="Get property details.",
        parameters={
            "type": "object",
            "properties": {
                "property_id": {"type": "string", "description": "The internal ID or listing ID of the property."},
            },
            "required": ["property_id"]
        }
    ),

    # 🔐 SECURITY
    ToolDefinition(
        name="generate_password",
        description="Generate secure password.",
        parameters={
            "type": "object",
            "properties": {
                "length": {"type": "number", "description": "The total number of characters the password should have."},
            },
            "required": ["length"]
        }
    ),
    ToolDefinition(
        name="check_password_strength",
        description="Check password strength.",
        parameters={
            "type": "object",
            "properties": {
                "password": {"type": "string", "description": "The password string to evaluate for strength and security."},
            },
            "required": ["password"]
        }
    ),

    # 📊 DATA / ANALYTICS
    ToolDefinition(
        name="run_sql_query",
        description="Run SQL query.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The exact SQL string to execute on the database."},
            },
            "required": ["query"]
        }
    ),
    ToolDefinition(
        name="generate_report",
        description="Generate report.",
        parameters={
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "The raw data, JSON, or metrics to compile into a report."},
            },
            "required": ["data"]
        }
    ),

    # 🧾 DOCUMENTS
    ToolDefinition(
        name="create_document",
        description="Create document.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the new document."},
                "content": {"type": "string", "description": "The body text to insert into the document."},
            },
            "required": ["title", "content"]
        }
    ),
    ToolDefinition(
        name="edit_document",
        description="Edit document.",
        parameters={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "The ID of the document to edit."},
                "content": {"type": "string", "description": "The new content to update the document with."},
            },
            "required": ["doc_id", "content"]
        }
    ),

    # 🧠 ADVANCED TASKS
    ToolDefinition(
        name="plan_trip",
        description="Plan a trip itinerary.",
        parameters={
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "The city or country you plan on traveling to."},
                "days": {"type": "number", "description": "The duration of the trip in days."},
            },
            "required": ["destination", "days"]
        }
    ),
    ToolDefinition(
        name="analyze_sentiment",
        description="Analyze sentiment of text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text snippet to evaluate for positive, neutral, or negative sentiment."},
            },
            "required": ["text"]
        }
    ),
]