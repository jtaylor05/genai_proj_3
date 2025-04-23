import os
import sys
import openai
from models import Models
from prompting_strategies import zero_shot_prompt, few_shot_prompt, chain_thought_prompt, self_consistency_prompt, role_play_prompt, prompt_chaining

models = ['chatgpt', 'codestral', 'gemini', 'claude', 'llama']
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
                    output.append(prompt_chaining(clients, clients.MODELS[model], prompt1, prompt2))
                    input_stream = input_stream[4:]
            except:
                raise IndexError("Not enough arguements for {} prompting".format(strat))

    out_file = os.path.join("data", "output", filename)

    with open(out_file, "w") as f:
        content = "\n@\n".join(output)
        f.write(content)
