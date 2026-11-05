from pathlib import Path
import gradio as gr
from agent_system import run, save_trace
import uuid
runtime=None

def handle(request, image):
    global runtime
    try:
        if runtime is None:
            from model_runtime import ModelRuntime
            lora=Path(__file__).resolve().parents[2]/'part2_diffusion/outputs/run1'
            runtime=ModelRuntime(lora)
        folder=Path(__file__).resolve().parents[1]/'outputs/app'/uuid.uuid4().hex
        state=run(request or '', image_path=image,vision_pipe=runtime,generation_pipe=runtime,
                  output_dir=folder)
        save_trace([state],folder/'trace.json')
    except (RuntimeError,OSError) as exc:
        return 'runtime_error',str(exc),'','No generation was performed.',None
    return state.decision, state.caption+'\nAnswer: '+state.visual_answer, state.prompt, '\n'.join(
        f"{x['agent']}: {x['output']}" for x in state.trace),state.output_path or None

demo=gr.Interface(fn=handle, inputs=[gr.Textbox(label='Request'),gr.Image(type='filepath',label='Optional image')], outputs=[gr.Textbox(label='Decision'),gr.Textbox(label='Caption and answer'),gr.Textbox(label='Prompt'),gr.Textbox(label='Trace'),gr.Image(label='Generated image')], title='DATA 266 Lab 2 Part 3')
if __name__=='__main__': demo.queue(default_concurrency_limit=1).launch()
