import argparse
import logging
from pathlib import Path
from src.data.diffusion_augmentation import augment_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Augment cutlery dataset using diffusion models"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Input directory containing original images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented",
        help="Output directory for augmented images",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=5,
        help="Number of variations to generate per image",
    )

    args = parser.parse_args()

    # Ensure input directory exists
    if not Path(args.input_dir).exists():
        raise ValueError(f"Input directory {args.input_dir} does not exist!")

    logging.info(f"Starting dataset augmentation...")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Variations per image: {args.variations}")

    # Run augmentation
    augment_dataset(
        input_root=args.input_dir,
        output_root=args.output_dir,
        variations_per_image=args.variations,
    )

    logging.info("Dataset augmentation completed!")


if __name__ == "__main__":
    main()
