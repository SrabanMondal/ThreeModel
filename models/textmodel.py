import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_obj
import os
def text3d(model,output,prompt):
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = ShapEPipeline.from_pretrained(model['text'])
    pipe = pipe.to(device)
    guidance_scale = model['guidance_scale']
    #prompt = input('Write prompt to generate 3d model')

    images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=model['num_inference_steps'],
            frame_size=model['frame_size'],
            output_type="mesh"
        ).images
    mesh_data = images[0]
    print("Decoding complete.")
    print(f"Exporting mesh to {output}{prompt}.obj")
    os.makedirs(output, exist_ok=True)
    export_to_obj(mesh_data, output+prompt.strip()+'.obj')
    print(f"OBJ file saved successfully: {output}{prompt}.obj")
    return output+prompt+'.obj'
    