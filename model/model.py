import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

repo_id = "Mochalka123/ZoltanChivayLLama-3.1-8B"
gguf_filename = "Meta-Llama-3.1-8B.Q8_0.gguf" 

model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, 
    n_ctx=2048,     
    verbose=False,
    chat_format="chatml"
)

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()
    
def generate_response(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({"role": "user", "content": message})

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.8,    
        top_p=0.95,
        repeat_penalty=1.1, 
        stream=True
    )
    
    partial_text = ""
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            partial_text += delta['content']
            yield partial_text

demo = gr.ChatInterface(
    fn=generate_response,
    type="messages",
    title="Zoltan Chivay Chat",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
