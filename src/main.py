from models import Models
import prompting_strategies

models = ['gpt-4o', 'Codestral-2501', 'gemini-2.0-flash', 'claude-3-7-sonnet-20250219']

clients = Models()

prompt = "Tell me about yourself"

print(clients.chatgpt(prompt).choices[0].message.content)
print(clients.codestral(prompt).choices[0].message.content)
print(clients.gemini(prompt).text)
print(clients.claude(prompt))