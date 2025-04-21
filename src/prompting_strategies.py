from models import Models
from nltk.translate.bleu_score import sentence_bleu

def zero_shot_prompt(models : Models, model, prompt):
    return models.request(model, models.to_message_format(model, prompt))
def few_shot_prompt(models : Models, model, prompt, examples):
    system_prompt = "\n".join(examples)
    return models.request(model, models.to_message_format(model, prompt, system_prompt))
def chain_thought_prompt(models : Models, model, prompt):
    prompt = prompt + "\n Explain your thought process step-by-step."
    return models.request(model, models.to_message_format(model, prompt))
def self_consistency_prompt(models : Models, model, prompt):
    responses = []
    for i in range(3):
        responses.append(models.request(model, models.to_message_format(model, prompt)))
    max_bleu = 0.0
    max_bleu_response = "No Response"
    for response in responses:
        references = [x for x in responses if x != response]
        current_bleu = sentence_bleu(references, response)
        if current_bleu > max_bleu:
            max_bleu = current_bleu
            max_bleu_response = response
    return max_bleu_response
def role_play_prompt(models : Models, model, prompt, role):
    return models.request(model, models.to_message_format(model, prompt, role), 1024)
def prompt_chaining(models : Models, model, prompt1, prompt2):
    conversation = models.to_message_format(model, prompt1)
    reply = models.request(model, conversation)
    conversation.append({"role": "assistant", "content": reply})
    conversation.append({"role": "user", "content": prompt2})
    return reply + "\n" + prompt2 + "\n" + models.request(model, conversation)
    