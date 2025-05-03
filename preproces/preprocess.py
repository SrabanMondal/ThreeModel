import cv2
import numpy as np
from PIL import Image, ImageFilter
import rembg
import matplotlib.pyplot as plt

try:
    cpu_session = rembg.new_session(model_name="u2net", providers=['CPUExecutionProvider'])
    print("Using CPU session for background removal.")
    REMGB_SESSION = cpu_session
except Exception as e:
    print(f"Failed to create rembg CPU session: {e}. Background removal might fail.")
    REMGB_SESSION = None

def apply_clahe_on_rgba(img_pil_rgba):
    if img_pil_rgba.mode != 'RGBA':
        print("Image is not RGBA")
        return img_pil_rgba
    
    R, G, B, A = img_pil_rgba.split()
    img_rgb_np = np.array(Image.merge("RGB", (R, G, B)))
    img_lab = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    final_rgb_np = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    final_rgb_pil = Image.fromarray(final_rgb_np)
    final_rgba = Image.merge("RGBA", (*final_rgb_pil.split(), A))
    return final_rgba

def preprocess_image_enhanced(
    image,
    target_size=256,
    session=REMGB_SESSION,
    clahe=True,       
    sharpening=True,  
    sharpen_radius=0.5,     
    padding=10              
    ):
    if not session:
        print("Background removing fail")
        try:
            print("Resizing image")
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
            else:
                 img = image.convert('RGB')
            img_resized = img.resize((target_size, target_size), Image.LANCZOS)
            print("Resizing done")
            return img_resized
        except Exception as error:
            print(f"Preprocess failed: {error}")
            return None
    try:
        if isinstance(image, str):
            img_pil = Image.open(image)
        elif isinstance(image, Image.Image):
            img_pil = image
        else:
            raise TypeError("No image")
        img_pil = img_pil.convert('RGB')
        print("Removing background")
        img_bg = rembg.remove(img_pil, session=session)
        print("Background remove done")
        img = img_bg
        if clahe:
            print("Applying clahe")
            img = apply_clahe_on_rgba(img)
            print("clahe done")

        if sharpening:
            print("Sharpening img")
            if img.mode == 'RGBA':
                R, G, B, A = img.split()
                img_rgb = Image.merge("RGB", (R, G, B))
                img_rgb_sharpened = img_rgb.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=150, threshold=3))
                img = Image.merge("RGBA", (*img_rgb_sharpened.split(), A))
            else:
                 img = img.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=150, threshold=3))
            print("img sharpened")

        if padding > 0:
            print(f"Adding {padding}px transparent padding")
            padded_size = (img.width + 2 * padding, img.height + 2 * padding)
            padded_image = Image.new("RGBA", padded_size, (0, 0, 0, 0))
            paste_position = (padding, padding)
            padded_image.paste(img, paste_position, mask=img)
            img = padded_image
            print("Padding done")

        print(f"Resizing image")
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        print("Resizing done")
        plt.imshow(img_resized)
        plt.axis('off')
        plt.show()
        return img_resized

    except FileNotFoundError:
        print(f"Error: Image file not found at {image}")
        return None
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        try:
            print("preprocess failed, resizing image")
            if isinstance(image, str):
                img_pil = Image.open(image).convert('RGB')
            else:
                 img_pil = image.convert('RGB')
            img_resized_fallback = img_pil.resize((target_size, target_size), Image.LANCZOS)
            print("Fallback resizing done")
            return img_resized_fallback
        except Exception as fallback_e:
            print(f"preprocess failed: {fallback_e}")
            return None