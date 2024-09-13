import torch
from model import Net
from utils import load_model
from torchvision import transforms
from PIL import Image

def predict_image(image_path, model_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    model = Net(num_classes=10)
    model = load_model(model, model_path)
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)
    return prediction.item()

if __name__ == '__main__':
    image_path = 'path_to_your_image.jpg'
    model_path = 'model.pth'
    prediction = predict_image(image_path, model_path)
    print(f'Predicted class index: {prediction}')
