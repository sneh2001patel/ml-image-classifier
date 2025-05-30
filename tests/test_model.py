import torch

from src.model import CNN


def test_model_forward():
    model = CNN()
    model.eval()

    # Create dummy input: batch of 4 images, 3 channels, 128x128 (same as training)
    dummy_input = torch.randn(4, 3, 128, 128)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Output should have shape (4, 2) since 2 classes (cats, dogs)
    assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"


def test_model_trainable():
    model = CNN()

    # Check that model parameters are trainable (require_grad == True)
    params = list(model.parameters())
    assert any(
        p.requires_grad for p in params
    ), "Model parameters should require gradients."
