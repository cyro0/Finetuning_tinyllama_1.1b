from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import transformers

#Load Model
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=False, revision="main")

#Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

#Use Base Model
model.eval()

comment = "Great content, thank you!"
prompt = f'''[INST] {comment} [INST]'''

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(input_ids=inputs["inputs_ids"].to("cuda"), max_new_tokens=140)

print(tokenizer.batch_decode(outputs)[0])

#Prompt Engineering
intstructions_string = f"""ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. \
ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""

prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

prompt = prompt_template(comment)
print(prompt)

