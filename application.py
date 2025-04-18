# application.py

import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# —— 1. Set Same CNN —— 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(7 * 7 * 64, 128)
        self.fc2   = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [batch,32,14,14]
        x = self.pool(F.relu(self.conv2(x)))   # [batch,64,7,7]
        x = x.view(x.size(0), -1)              # [batch,7*7*64]
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# —— 2. Load PTH —— 
MODEL_PATH = 'mnist_cnn.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Can't Find:{MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# —— 3. Load Tkinter Window —— 
root = tk.Tk()
root.title("Hand Writtern Digits Recognization")
result_var = tk.StringVar(value="Test Result:")

# Set Canvas Size 280×280
canvas_size = 280
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
canvas.pack(padx=10, pady=10)

# Save PIL
image1 = Image.new("L", (canvas_size, canvas_size), 'white')
draw   = ImageDraw.Draw(image1)

# —— 4. Printing —— 
def paint(event):
    x, y = event.x, event.y
    r = 8   # pen r
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
    draw.ellipse([x-r, y-r, x+r, y+r], fill='black')

canvas.bind("<B1-Motion>", paint)

def preprocess(pil_img):
    # Decrease back to 28×28，Using LANCZOS
    img28 = pil_img.resize((28, 28), resample=Image.Resampling.LANCZOS)
    # 反色
    img28 = ImageOps.invert(img28)
    # Normalization Using Numpy [0,1]
    arr = np.array(img28, dtype=np.float32) / 255.0
    # Normalization
    arr = (arr - 0.1307) / 0.3081
    # 转 Tensor
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    return tensor

# —— 6. Perdict —— 
def predict():
    x = preprocess(image1)
    with torch.no_grad():
        out = model(x)
    pred = out.argmax(dim=1).item()
    result_var.set(f"Test Result:{pred}")

# —— 7. Cleaning —— 
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,canvas_size,canvas_size], fill='white')
    result_var.set("Test Result:")

# —— 8. Buttons and Labels —— 
btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Test", command=predict).pack(side='left', padx=5)
tk.Button(btn_frame, text="Clear", command=clear_canvas).pack(side='left', padx=5)
tk.Label(root, textvariable=result_var, font=('Arial', 16)).pack(pady=10)

# —— 9. Start loop —— 
root.mainloop()
