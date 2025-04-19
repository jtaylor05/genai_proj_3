import os
from openai import OpenAI
from google import genai
import anthropic

class Models:

    def __init__(self):
        with open("data/anthropic_token.txt", "r") as f:
            os.environ["ANTH_TOKEN"] = f.read().strip()
        self.anthropic_client = anthropic.Anthropic(api_key = os.environ["ANTH_TOKEN"]) 
        with open("data/gemini_token.txt", "r") as f:
            os.environ["GEM_TOKEN"] = f.read().strip()
        self.gemini_client = genai.Client(api_key=os.environ["GEM_TOKEN"])
        with open("data/openai_token.txt", "r") as f:
            os.environ["OPEN_TOKEN"] = f.read().strip()
        self.openai_client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["OPEN_TOKEN"])

    def claude(self, user_prompt, system_prompt, max_tokens):
        message = self.anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": user_prompt
                }
                ]
        )
        return message
    
    def gemini(self, user_prompt, system_prompt, max_tokens):
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash", 
            contents= " ".join([system_prompt, user_prompt])
        )
        return response
    
    def chatgpt(self, user_prompt, system_prompt, max_tokens):
        response = self.openai_client.responses.create(
            model="gpt-4.1",
            max_output_tokens=max_tokens,
            input=[
                {
                    "role": "developer",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        return response
    
    def codestral(self, user_prompt, system_prompt, max_tokens):    
        response = self.openai_client.responses.create(
            model="Codestral-2501",
            max_output_tokens=max_tokens,
            input=[
                {
                    "role": "developer",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        return response    
    
    def reset(self):
        self.anthropic_client = anthropic.Anthropic(api_key = os.environ["ANTH_TOKEN"]) 
        self.gemini_client = genai.Client(api_key=os.environ["GEM_TOKEN"])
        self.openai_client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["OPEN_TOKEN"])