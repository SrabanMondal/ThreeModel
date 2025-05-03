import torch
from PIL import Image
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_obj
from preproces import preprocess_image_enhanced
import os
#image = preprocess_image('/kaggle/input/3dtest/cup.png')
#image = Image.open('/kaggle/input/3dtest/cup.png').resize((256, 256))
#image = preprocess_image_rembg(IMAGE_PATH, target_size=TARGET_SIZE)
def image3d(model,image,output,path, clahe, sharp, pad):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = ShapEImg2ImgPipeline.from_pretrained(model['image']).to(device)
        img_size = image['size']
        guidance_scale = model['guidance_scale']
        image = preprocess_image_enhanced(
        path,
        target_size=img_size,
        clahe=clahe,       
        sharpening=sharp,
        padding=pad         
        )
        result = pipe(
        image,
        guidance_scale=guidance_scale,
        num_inference_steps=model['num_inference_steps'],
        frame_size=model['frame_size'],
        output_type="mesh"
        )
        mesh_data = result.images[0]
        os.makedirs(output, exist_ok=True)
        print(f"Exporting mesh to {output}temp.obj")
        export_to_obj(mesh_data, output+'temp.obj')
        print(f"OBJ file saved successfully: {output}temp.obj")
        return output+'temp.obj'
