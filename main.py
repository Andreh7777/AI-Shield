from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import Anonymize, PromptInjection, TokenLimit, Toxicity
from llm_guard.output_scanners import Deanonymize, NoRefusal, Relevance, Sensitive
from llm_guard.vault import Vault
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from configparser import ConfigParser

# Initialize ConfigParser to read from the configuration file
config = ConfigParser()
config.read('config.ini')

# Initialize the Vault to securely store and retrieve sensitive data
vault = Vault()

# Define a list of input scanners to process and sanitize incoming prompts
input_scanners = [Anonymize(vault), Toxicity(), TokenLimit(), PromptInjection()]

# Define a list of output scanners to process and sanitize the model's output before returning it to the user
output_scanners = [Deanonymize(vault), NoRefusal(), Relevance(), Sensitive()]

# Initialize the FastAPI application
app = FastAPI()

# Define a Pydantic model for the incoming request payload
class ChatRequest(BaseModel):
    prompt: str

# Asynchronous function to interact with the external AI model API
async def get_response(prompt):
    # Get the API URL from the config file
    url = config.get('DEFAULT', 'api_url')
    
    # Set up headers including the Authorization token from the config
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f"Bearer {config.get('DEFAULT', 'JWT_TOKEN')}"
    }
    
    # Prepare the payload with the model and the user prompt
    payload = {
        "model": config.get('DEFAULT', 'model'),
        "messages": [{"role": "user", "content": prompt}]
    }

    # Use an async HTTP client to send a POST request to the API
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error if the request was unsuccessful
        return response.json()  # Return the JSON response from the API

# Define the endpoint to handle chat completions
@app.post("/v1/chat/completions")
async def generate_response(request: ChatRequest):
    # Scan and sanitize the incoming prompt using the input scanners
    sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, request.prompt)
    
    # If any of the scanners indicate the prompt is invalid, raise a 400 HTTP error
    if any(results_valid.values()) is False:
        raise HTTPException(status_code=400, detail=f"Prompt not valid, scores: {results_score}")
    
    # Call the external API to generate a response based on the sanitized prompt
    response_data = await get_response(sanitized_prompt)
    response_text = response_data['choices'][0]['message']['content']  
    
    # Scan and sanitize the API's response using the output scanners
    sanitized_response_text, results_valid, results_score = scan_output(output_scanners, sanitized_prompt, response_text)
    
    # If any of the scanners indicate the output is invalid, raise a 400 HTTP error
    if any(results_valid.values()) is False:
        raise HTTPException(status_code=400, detail=f"Output not valid, scores: {results_score}")
    
    # Return the sanitized prompt and response as a JSON object
    return {"prompt": sanitized_prompt, "response": sanitized_response_text}
