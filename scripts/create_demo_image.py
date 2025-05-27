#!/usr/bin/env python3
"""
Create Demo Image for Testing Inference Pipeline

This script creates a simple demo image that can be used to test
the inference pipeline when no real cutlery images are available.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path


def create_demo_cutlery_image(output_path: str, cutlery_type: str = "fork"):
    """
    Create a simple demo image of cutlery for testing.

    Args:
        output_path: Path to save the demo image
        cutlery_type: Type of cutlery to draw ('fork', 'knife', 'spoon')
    """
    # Create a 320x320 image with white background
    img = Image.new("RGB", (320, 320), color="white")
    draw = ImageDraw.Draw(img)

    # Draw a simple representation of cutlery
    if cutlery_type == "fork":
        # Draw fork handle
        draw.rectangle([150, 50, 170, 250], fill="gray")

        # Draw fork prongs
        for i in range(4):
            x = 140 + i * 10
            draw.rectangle([x, 50, x + 5, 120], fill="gray")

        # Add text label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((100, 280), "Demo Fork", fill="black", font=font)

    elif cutlery_type == "knife":
        # Draw knife handle
        draw.rectangle([150, 150, 170, 270], fill="brown")

        # Draw knife blade
        draw.polygon([(160, 50), (180, 150), (140, 150)], fill="silver")

        # Add text label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((100, 280), "Demo Knife", fill="black", font=font)

    elif cutlery_type == "spoon":
        # Draw spoon handle
        draw.rectangle([150, 120, 170, 270], fill="gray")

        # Draw spoon bowl
        draw.ellipse([130, 50, 190, 130], fill="gray")

        # Add text label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((100, 280), "Demo Spoon", fill="black", font=font)

    # Add some noise to make it more realistic
    img_array = np.array(img)
    noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
    img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    # Save the image
    img.save(output_path)
    print(f"Demo {cutlery_type} image saved: {output_path}")


def main():
    """Create demo images for all cutlery types."""
    output_dir = Path("demo_images")
    output_dir.mkdir(exist_ok=True)

    # Create demo images for each type
    cutlery_types = ["fork", "knife", "spoon"]

    for cutlery_type in cutlery_types:
        output_path = output_dir / f"demo_{cutlery_type}.jpg"
        create_demo_cutlery_image(str(output_path), cutlery_type)

    print(f"\n✅ Demo images created in {output_dir}/")
    print("You can now test inference with:")
    print(
        f"  python scripts/infer_image.py --model <model_path> --image {output_dir}/demo_fork.jpg --visualize"
    )


if __name__ == "__main__":
    main()
