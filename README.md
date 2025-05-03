# 🧱 ThreeModel: Text/Image to 3D Model Generator

**ThreeModel** is a Gradio-based application that takes either **text** or **image** input and generates a 3D shape using **Hugging Face Diffusers**. This project allows you to interact with powerful 3D generative models through a simple web interface or directly via the terminal.

---

## ✨ Features

- 🔤 Text-to-3D generation  
- 🖼️ Image-to-3D generation  
- 💻 Gradio web UI for easy interaction  
- 🛠️ Terminal-friendly usage with function calls  
- 📊 Built-in visualization using Gradio and Matplotlib  

---

## 🚀 Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/SrabanMondal/ThreeModel.git
cd ThreeModel
```
### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the Application
```sh
python -u main.py
```

After running, you will receive a Gradio URL in the terminal.
Open the URL in your browser to start generating 3D models.

## Terminal Mode (No Gradio)
If you prefer not to use the Gradio UI:
-Remove the app.launch() or related Gradio interface lines from main.py.
-Call the functions directly in your own script or Python shell.
-For visualization:
Use the utility function included in the project to visualize results with Matplotlib.
Or refer to Gradio's logic to customize your own output view.

![3D Model](https://drive.google.com/uc?export=view&id=)