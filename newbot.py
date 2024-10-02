from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

def predict(input, history=[]):
    instruction = 'Instruction: given a dialog context related to urban living and city information, you need to respond informatively and engagingly'
    knowledge = '  '
    s = list(sum(history, ()))
    s.append(input)
    dialog = ' EOS '.join(s)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    top_p = 0.9
    min_length = 8
    max_length = 64

    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(f"{query}", return_tensors='pt')

    output = model.generate(new_user_input_ids, min_length=min_length, max_length=max_length, top_p=top_p, do_sample=True).tolist()
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    history.append((input, response))
    return response, history

import gradio as gr
gr.Interface(fn=predict,
             inputs=["text", 'state'],
             outputs=["text", 'state']).launch(debug=True, share=True)
