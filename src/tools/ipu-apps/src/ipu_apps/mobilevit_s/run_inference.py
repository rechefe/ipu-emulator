import ipu
from mobilevit_s import MobileViT_S

# Build model and load pre-trained weights
model = MobileViT_S(num_classes=1000)
model.load_weights("weights/")

# Load and preprocess input image
image = ipu.load_image("cat.jpg", size=(256, 256))

# Run inference on IPU
output = model(image)

# Display prediction
class_id = output.argmax()
print(f"Predicted: {ipu.imagenet_labels[class_id]}")
