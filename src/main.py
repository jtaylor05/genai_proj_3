import os
import sys
from models import Models
from prompting_strategies import zero_shot_prompt, few_shot_prompt, chain_thought_prompt, self_consistency_prompt, role_play_prompt, prompt_chaining

models = ['chatgpt', 'codestral', 'gemini', 'claude']
strats = ['ZS', 'FS', 'COT', 'SC', 'RP', 'PC']

clients = Models()

files_to_check = [int(x) for x in sys.argv[1:]]

for i in files_to_check:
    output = []
    filename = "task" + str(i) + ".txt"
    in_file = os.path.join("data", "input", filename)
    with open(in_file, "r") as f:
        input_stream = f.read().split("@")
        while len(input_stream) > 2:
            model = input_stream[0].strip()
            strat = input_stream[1].strip()
            print("Model: {}, Strategy: {}".format(model, strat))
            try:
                if strat == 'ZS':
                    prompt = input_stream[2]
                    output.append(model + "\n")
                    output.append(strat + "\n")
                    output.append(zero_shot_prompt(clients, clients.MODELS[model], prompt))
                    input_stream = input_stream[3:]
                if strat == 'FS':
                    prompt1 = input_stream[2]
                    prompt2 = input_stream[3]
                    output.append(model + "\n")
                    output.append(strat + "\n")
                    output.append(few_shot_prompt(clients, clients.MODELS[model], prompt2, prompt1))
                    input_stream = input_stream[4:]
                if strat == 'COT':
                    prompt = input_stream[2]
                    output.append(model + "\n")
                    output.append(strat + "\n")
                    output.append(chain_thought_prompt(clients, clients.MODELS[model], prompt))
                    input_stream = input_stream[3:]
                if strat == 'SC':
                    prompt = input_stream[2]
                    output.append(model + "\n")
                    output.append(strat + "\n")
                    output.append(self_consistency_prompt(clients, clients.MODELS[model], prompt))
                    input_stream = input_stream[3:]
                if strat == 'RP':
                    prompt1 = input_stream[2]
                    prompt2 = input_stream[3]
                    output.append(model + "\n")
                    output.append(strat + "\n")
                    output.append(role_play_prompt(clients, clients.MODELS[model], prompt2, prompt1))
                    input_stream = input_stream[4:]
                if strat == 'PC':
                    prompt1 = input_stream[2]
                    prompt2 = input_stream[3]
                    output.append(model + "\n")
                    output.append(strat + "\n")
                    output.append(role_play_prompt(clients, clients.MODELS[model], prompt1, prompt2))
                    input_stream = input_stream[4:]
            except:
                raise IndexError("Not enough arguements for {} prompting".format(strat))

    out_file = os.path.join("data", "output", filename)

    with open(out_file, "w") as f:
        content = "\n@\n".join(output)
        f.write(content)
# prompt = "Tell me about yourself"

# print("CHATGPT: \n", clients.chatgpt(prompt).choices[0].message.content)
# print("CODESTRAL: \n", clients.codestral(prompt).choices[0].message.content)
# print("GEMINI: \n", clients.gemini(prompt).text)
# #print(clients.claude(prompt))

# print("0-SHOT:\n", zero_shot_prompt(clients, clients.MODELS['chatgpt'], "Write me a recipe for deviled eggs that can feed 20 people."))
# print("1-SHOT:\n", few_shot_prompt(clients, clients.MODELS['chatgpt'], "Could you please summarize translate the following sentence into German: Hi, my name is Jackson. Translation:", "Examples of translation: That is a dog. Translation: Das ist ein Hund"))
# print("COT:\n", chain_thought_prompt(clients, clients.MODELS['chatgpt'], "Could you please describe to me how a volcano explodes?"))
# print("Self-Consistency:\n", self_consistency_prompt(clients, clients.MODELS['chatgpt'], "Why did Peter Piper picked a pinch of pickled peppers?"))
# print("Role-Play:\n", role_play_prompt(clients, clients.MODELS['chatgpt'], "How does one reduce taxes?", "Act as a professional financial advisor for the user."))
# print("Prompt-Chaining:\n", prompt_chaining(clients, clients.MODELS['chatgpt'], "Tell me about the people of the Himalayas", "Could you repeat what you said but in a British Accent"))