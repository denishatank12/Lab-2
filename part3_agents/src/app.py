from pathlib import Path
import gradio as gr
from agent_system import run

def handle(request, image):
    state=run(request, image_path=image)
    return state.decision, state.caption, state.prompt, '\n'.join(f"{x['agent']}: {x['output']}" for x in state.trace)

demo=gr.Interface(fn=handle, inputs=[gr.Textbox(label='Request'),gr.Image(type='filepath',label='Optional image')], outputs=[gr.Textbox(label='Decision'),gr.Textbox(label='Caption'),gr.Textbox(label='Prompt'),gr.Textbox(label='Trace')], title='DATA 266 Lab 2 Part 3')
if __name__=='__main__': demo.launch()
