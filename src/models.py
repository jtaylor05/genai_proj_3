import os
from openai import OpenAI
from google import genai
import anthropic

class Models:

    MODELS = {
                'chatgpt':'gpt-4o',
                'codestral':'Codestral-2501', 
                'gemini':'gemini-2.0-flash', 
                'claude':'claude-3-7-sonnet-20250219',
                'llama':'Llama-4-Maverick-17B-128E-Instruct-FP8'
             }

    def __init__(self):
        with open("data/anthropic_token.txt", "r") as f:
            os.environ["ANTH_TOKEN"] = f.read().strip()
        self.anthropic_client = anthropic.Anthropic(api_key = os.environ["ANTH_TOKEN"]) 
        with open("data/gemini_token.txt", "r") as f:
            os.environ["GEM_TOKEN"] = f.read().strip()
        self.gemini_client = genai.Client(api_key=os.environ["GEM_TOKEN"])
        with open("data/github_token.txt", "r") as f:
            os.environ["OPEN_TOKEN"] = f.read().strip()
        self.openai_client = OpenAI(
                                    #base_url="https://api.openai.com/v1/chat/completions",
                                    base_url="https://models.inference.ai.azure.com", 
                                    api_key=os.environ["OPEN_TOKEN"])

    def request(self, model, messages, max_tokens = 1024):
        if model == self.MODELS['chatgpt']:
            return self.chatgpt(messages, max_tokens)
        if model == self.MODELS['codestral']:
            return self.codestral(messages, max_tokens)
        if model == self.MODELS['gemini']:
            return self.gemini(messages, max_tokens)
        if model == self.MODELS['claude']:
            return self.claude(messages, max_tokens)
        if model == self.MODELS['llama']:
            return self.llama(messages, max_tokens)
        
        raise Exception("Input model does not exist")
    
    def to_message_format(self, model, user_prompt, system_prompt = ""):
        if model == self.MODELS['chatgpt'] or model == self.MODELS['codestral'] or model == self.MODELS['claude'] or model == self.MODELS['llama']:
            if system_prompt == "":
                return [{ "role": "user", "content": user_prompt }]
            return [{ "role": "system", "content": system_prompt },
                    { "role": "user", "content": user_prompt }]
        if model == self.MODELS['gemini']:
            return " ".join([system_prompt, user_prompt])
        
    def add_to_message_formate(self, model, message, user_prompt, assistant_reply):
        if model == self.MODELS['chatgpt'] or model == self.MODELS['codestral'] or model == self.MODELS['claude'] or model == self.MODELS['llama']:
            message.append({"role": "assistant", "content": assistant_reply})
            message.append({"role": "user", "content": user_prompt})
            return message
        if model == self.MODELS['gemini']:
            return " ".join([message, assistant_reply, user_prompt])

    def claude(self, messages, max_tokens = 1024):
        message = self.anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=max_tokens,
            # system=system_prompt,
            messages=messages
        )
        return message
    
    def llama(self, messages, max_tokens = 1024):
        response = self.openai_client.chat.completions.create(
            model=self.MODELS['llama'],
            max_tokens=max_tokens,
            messages=messages
        )
        return response.choices[0].message.content

    def gemini(self, messages, max_tokens = 1024):
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash", 
            contents= messages
        )
        return response.text
    
    def chatgpt(self, messages, max_tokens = 1024):
        response = self.openai_client.chat.completions.create(
            model="gpt-4.1",
            max_tokens=max_tokens,
            messages=messages
        )
        return response.choices[0].message.content
    
    def codestral(self, messages, max_tokens = 1024):    
        response = self.openai_client.chat.completions.create(
            model="Codestral-2501",
            max_tokens=max_tokens,
            messages=messages
        )
        return response.choices[0].message.content