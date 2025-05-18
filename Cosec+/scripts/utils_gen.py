def gen_prompt(prompt: str, model_path:str) -> str:
    if "starcoder" in model_path.lower():
        # prompt=prompt+"\n    # Write your code here"
        prompt = "<fim_prefix>" + prompt + "<fim_suffix><fim_middle>" # 加跟不加\n
        # print(prompt)
    if "starchat" in model_path.lower():
        prompt = f"<|system|>\n<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>"
    return prompt
def replace_gen_prompt(prompt: str, model_path:str) -> str:
    if "starcoder" in model_path.lower():
        prompt = prompt.replace("<fim_prefix>","")
        prompt = prompt.replace("<fim_suffix><fim_middle>","")
        
    else:
        prompt = prompt.replace("Please complete the following Python code without providing any additional tasks such as testing or explanations\n","")
    if "starchat" in model_path.lower():
        prompt = prompt.replace("<|system|>\n<|end|>\n<|user|>","")
        prompt = prompt.replace("<|end|>\n<|assistant|>","")
    return prompt