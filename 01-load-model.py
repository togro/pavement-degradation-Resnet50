import torch
# from torchvision import models, transforms
from torchvision import transforms 
from PIL import Image
import torch.nn.functional as F

model = torch.load('./model/model-APD-pytorch-res50-92accuracy.pt', map_location='cpu')

model.eval()  # Cambia el modelo a modo de evaluación

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar y transformar la imagen
image = Image.open('/tmp/fatigue_crack (1073).jpeg')
image = Image.open('./xx.jpg')
image = transform(image)
image = image.unsqueeze(0)  # Añade una dimensión de batch al principio

# with torch.no_grad():  # Desactiva el cálculo de gradientes para la inferencia
#     outputs = model(image)
#     _, predicted = torch.max(outputs, 1)


# print("Índice de la clase predicha:", predicted.item())
# # Asumiendo que el orden alfabético determina los índices:
# clases = ["Fatigue cracks", "Linear cracks", "Potholes"]

# # Imprimir el nombre de la clase predicha
# predicted_class = clases[predicted.item()]
# print("Clase predicha:", predicted_class)

# Realizar la inferencia
with torch.no_grad():
    outputs = model(image)
    probabilities = F.softmax(outputs, dim=1)
    predicted_prob, predicted_index = torch.max(probabilities, 1)

# Asumiendo el orden alfabético para los nombres de las clases
clases = ["Fatigue cracks", "Linear cracks", "Potholes"]
predicted_class = clases[predicted_index.item()]
print(f"Clase predicha: {predicted_class}, Probabilidad: {predicted_prob.item():.4f}")