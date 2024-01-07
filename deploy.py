import torch
from torchvision import transforms
from PIL import Image
from train import ResNetClassifier


# Function to load the model from a checkpoint
def load_model_from_checkpoint(checkpoint_path, num_classes):
    model = ResNetClassifier.load_from_checkpoint(
        checkpoint_path, num_classes=num_classes
    )
    model.eval()  # Set the model to evaluation mode
    return model


# Function to perform inference on a single image
def predict_image(model, image_path):
    # Define the same transformations as used during training
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Open the image, apply transformations and add batch dimension
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        logits = model(image)
        predictions = torch.argmax(logits, dim=1)
    return predictions


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "/Users/neo/Documents/Research/DeepHeme/LLResults/LLSpecimenClassifier/lightning_logs/version_0/checkpoints/epoch=99-step=1600.ckpt"
    num_classes = 4  # Set the correct number of classes
    model = load_model_from_checkpoint(checkpoint_path, num_classes)

    image_path = (
        "/media/hdd3/neo/topviews_1k/BMA/H22-10251_S11_MSKY_2023-06-12_18.55.56.png"
    )
    prediction = predict_image(model, image_path)
    print(f"Predicted class: {prediction.item()}")
