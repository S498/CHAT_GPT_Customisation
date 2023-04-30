import os
import openai
import torch
import fitz
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
from flask import Flask, render_template, request, jsonify

openai.api_key = "sk-"
app = Flask(__name__)

gpt3_davinci_engine = "text-davinci-002"
gpt3_generation_temperature = 0.5
gpt3_max_response_tokens = 1024

t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_summarization_model = T5ForConditionalGeneration.from_pretrained("t5-base")

t5_summarization_model.eval()
t5_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_summarization_model.to(t5_device)

def generate_gpt3_answer(prompt):
    response = openai.Completion.create(
        engine=gpt3_davinci_engine,
        prompt=prompt,
        max_tokens=gpt3_max_response_tokens,
        n=1,
        stop=None,
        temperature=gpt3_generation_temperature,
        timeout=60,
    )
    return response.choices[0].text.strip()

def extract_pdf_text(file_path):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def generate_question_prompts_from_summary(summary):
    prompts = [f"{sentence.strip()}?" for sentence in summary.split(". ")]
    return prompts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatgpt', methods=['POST'])
def chatbot():
    user_input_text = request.json['prompt']
    handbook_path = 'docs/california_driver_handbook.pdf'
    handbook_text = extract_pdf_text(handbook_path)
    summary_ids = t5_tokenizer.encode(handbook_text, max_length=1024, truncation=True, return_tensors='pt').to(t5_device)
    summary = t5_summarization_model.generate(summary_ids, max_length=300, min_length=30, do_sample=False)
    summary = t5_tokenizer.decode(summary[0], skip_special_tokens=True)
    prompts = generate_question_prompts_from_summary(summary)
    response = generate_gpt3_answer("\n".join(prompts) + f"\nUser: {user_input_text}\nBot:")
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5001, debug=True)