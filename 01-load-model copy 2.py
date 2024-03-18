import torch
# from torchvision import models, transforms
from torchvision import transforms 
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# Asegúrate de definir el modelo exactamente como lo hiciste antes de guardarlo
# model = models.resnet50(pretrained=False)  # Si usaste ResNet50
# Cargar el modelo preentrenado con la nueva API
model = resnet50(weights=ResNet50_Weights.DEFAULT)

#
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 3)  # Asegúrate de ajustar el número de clases

# Cargar los pesos del modelo
model.load_state_dict(torch.load('./model/model-APD-pytorch-res50-92accuracy.pt'))
model.eval()  # Cambia el modelo a modo de evaluación

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar y transformar la imagen
image = Image.open('/tmp/atigue_crack (1073).jpeg')
image = transform(image)
image = image.unsqueeze(0)  # Añade una dimensión de batch al principio

with torch.no_grad():  # Desactiva el cálculo de gradientes para la inferencia
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
