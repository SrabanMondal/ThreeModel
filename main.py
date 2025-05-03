import yaml
from models import text3d
from models import image3d
from visualize import visualize
import gradio as gr
def load_config(path="./config/app.config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
def pipeline(choice,prompt,path):
    config = load_config()
    if(choice=='Text Prompt'):
        path = text3d(config['model'],config['output'],prompt)
    else:
        path = image3d(config['model'],config['image'],config['output'],path)
    visualize(path)
    return path
    
app = gr.Interface(
    fn = pipeline,
    inputs=[
        gr.Radio(choices=["Text Prompt","Image Prompt"], label='Choose any one option'),
        gr.Textbox(''),
        gr.Image(type='filepath', label='Enter image if u picked image prompt')
    ],
    outputs=gr.Model3D()
)
app.launch(share=True)
