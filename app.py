#Flask 網站
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import io
import os

app = Flask(__name__)

# ======================
# 1. 類別名稱
# ======================
classes = ["cardboard", "glass", "metal", "paper", "plastic"]

# ======================
# 2. CNN 模型（一定要跟 train 一樣）
# ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ======================
# 3. 載入模型
# ======================
model = CNN()
model.load_state_dict(torch.load("trash_cnn.pth", map_location="cpu"))
model.eval()

# ======================
# 4. 圖片前處理
# ======================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# ======================
# 5. 首頁
# ======================
@app.route("/")
def home():
    return render_template("index.html")

# ======================
# 6. AI 預測 API
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")

        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)

            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        return jsonify({
            "result": classes[pred.item()],
            "confidence": round(confidence.item() * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================
# 7. 啟動網站
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)