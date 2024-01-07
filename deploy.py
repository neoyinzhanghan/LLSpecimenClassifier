import torch
from torchvision import transforms
from PIL import Image
from train import ResNetClassifier

class_dct = {
    0: "BMA",
    1: "MPBorIBMA",
    2: "Others",
    3: "PB",
}


# Function to load the model from a checkpoint
def load_model_from_checkpoint(checkpoint_path, num_classes):
    model = ResNetClassifier.load_from_checkpoint(
        checkpoint_path, num_classes=num_classes
    )
    model.eval()  # Set the model to evaluation mode
    return model


def predict_image(model, image_path):
    # Define the same transformations as used during training
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x[:3, :, :]
            ),  # Keep only the first three channels
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Open the image, apply transformations and add batch dimension
    image = Image.open(image_path).convert("RGB")  # Convert image to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image tensor to the same device as the model
    device = next(model.parameters()).device  # Get the device of the model
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(image)
        predictions = torch.argmax(logits, dim=1)
    return class_dct[predictions.item()]


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "/home/greg/Documents/neo/LLSpecimenClassifier/lightning_logs/version_0/checkpoints/epoch=99-step=1600.ckpt"
    num_classes = 4  # Set the correct number of classes
    model = load_model_from_checkpoint(checkpoint_path, num_classes)

    image_path = (
        "/media/hdd3/neo/topviews_1k/Others/H21-5644_S10_MSK6_2023-05-23_17.55.11.png"
    )

    prediction = predict_image(model, image_path)
    print(f"Predicted class: {prediction}")