import os

import torch
import PIL.Image

from PIL import ImageDraw
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from functions.datasets import CocoDetectionWithFilenames


def draw_rectangle(
        orig: PIL.Image.Image,
        coords: tuple[int, int, int, int],
        new_path: str,
        rect_color='red',
        rect_width=1) -> None:
    copy = orig.copy()
    draw = ImageDraw.Draw(copy)

    draw.rectangle(coords, outline=rect_color, width=rect_width)
    copy.save(new_path)


def draw_bounding_boxes(image: PIL.Image.Image, box: tuple[int, int, int, int]) -> PIL.Image.Image:
    draw = ImageDraw.Draw(image)

    draw.rectangle(box, outline="black", width=2)

    return image


def save_images_with_bboxes(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    dataset: CocoDetectionWithFilenames,
    output_dir_category: str
) -> None:
    output_dir = os.path.join("data", output_dir_category)

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)

            for i in range(images.size(0)):
                image_tensor = images[i].cpu()
                image = to_pil_image(image_tensor)
                filename = dataset.get_filename(idx * images.size(0) + i)
                predicted_box = outputs[i].cpu().tolist()
                predicted_box = [int(num) for num in predicted_box]
                image_with_box = draw_bounding_boxes(image, predicted_box)

                image_with_box.save(os.path.join(output_dir, filename))

        print(f"Images with bounding boxes have been saved to {output_dir}")
