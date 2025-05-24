from PIL import Image

def get_combined_image(image_list, spacing=10, background_color=(255, 255, 255)):
    """
    Args：
    - image_list:  PIL.Image List
    - spacing: 10
    - background_color: (255, 255, 255)

    Return：
    - combined_image: single PIL.Image object
    """
    num_images = len(image_list)
    if num_images == 0:
        raise ValueError("The list is empty!")

    widths = [img.width for img in image_list]
    heights = [img.height for img in image_list]
    max_width = max(widths)
    total_height = sum(heights) + spacing * (num_images - 1)

    combined_image = Image.new('RGB', (max_width, total_height), color=background_color)

    y_offset = 0
    for img in image_list:
        x_offset = (max_width - img.width) // 2
        combined_image.paste(img, (x_offset, y_offset))
        y_offset += img.height + spacing  

    return combined_image