import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ===== 模型結構（要跟train一樣）=====
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
        return self.classifier(self.features(x))


# ===== 載入模型 =====
model = CNN()
model.load_state_dict(torch.load("trash_cnn.pth", map_location="cpu"))
model.eval()

# ===== image transform =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])
classes = ["cardboard", "glass", "metal", "paper", "plastic"]
# ===== 測試圖片 =====
img_path = "images/plastic.jpg"
img = Image.open(img_path).convert("RGB")

img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)

    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, dim=1)

pred_class = classes[pred.item()]
confidence = confidence.item() * 100

print("Prediction:", pred_class)
print("Confidence:", round(confidence, 2), "%")