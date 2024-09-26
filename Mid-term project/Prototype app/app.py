import gradio as gr
from functions import *

rag_poc = gr.ChatInterface(get_response)

if __name__ == "__main__":
    rag_poc.launch()