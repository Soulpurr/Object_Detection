"""Draw predicted or ground truth boxes on input image."""
import imghdr
import colorsys
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import backend as K

from functools import reduce
def ndarray_to_pil(image_np):
    """Convert a NumPy array to a PIL Image."""
    # Ensure the image is in uint8 format and scaled to [0, 255]
    image_np = (image_np * 255).astype(np.uint8)
    # Convert NumPy array to PIL Image
    image_pil = Image.fromarray(image_np)
    return image_pil
def preprocess_image_realtime(image, model_image_size):
    """
    Preprocess the input image for model prediction.

    Args:
        image: An OpenCV image array (BGR format).
        model_image_size: Tuple specifying the size (height, width) the image should be resized to.

    Returns:
        Tuple of (original_image, preprocessed_image_data).
    """
    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    resized_image = cv2.resize(image, (model_image_size[1], model_image_size[0]))  # model_image_size is (height, width)
    
    # Normalize image data
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.0
    
    # Add batch dimension
    image_data = np.expand_dims(image_data, axis=0)
    
    return image, image_data

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0] * 1.0
    width = image_shape[1] * 1.0
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    #image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score = scores.numpy()[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)

def draw_boxes1(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    #image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))
    image=ndarray_to_pil(image)
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score = scores.numpy()[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)


def draw_boxes_realtime(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image using OpenCV.

    Args:
        image: An `array` of shape (height, width, 3) with values in [0, 255] for uint8.
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indices into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")

    # Ensure image is in uint8 format and is in [0, 255]
    if np.max(image) > 255:
        image = (image * 255).astype(np.uint8)

    thickness = max((image.shape[0] + image.shape[1]) // 300, 1)  # Ensure thickness is at least 1
    colors = get_colors_for_classes1(len(class_names))

    for i, c in enumerate(box_classes):
        box_class = class_names[c]
        box = boxes[i]

        if scores is not None:
            score = scores[i]
            label = f'{box_class} {score:.2f}'
        else:
            label = box_class

        top, left, bottom, right = box
        top = max(0, int(np.floor(top + 0.5)))
        left = max(0, int(np.floor(left + 0.5)))
        bottom = min(image.shape[0], int(np.floor(bottom + 0.5)))
        right = min(image.shape[1], int(np.floor(right + 0.5)))

        # Debugging print statements
        print(f"Drawing box: {label}")
        print(f"Coordinates: top={top}, left={left}, bottom={bottom}, right={right}")

        # Ensure colors are integers and formatted correctly
        color = tuple(colors[c].tolist())
        if len(color) != 3:
            raise ValueError(f"Color for class {c} is not a valid RGB/BGR tuple: {color}")

        # Draw the bounding box
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

        # Put the label text above the bounding box
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_origin = (left, top - label_size[1]) if top - label_size[1] >= 0 else (left, top + 1)

        cv2.rectangle(image, (label_origin[0], label_origin[1] - label_size[1]),
                      (label_origin[0] + label_size[0], label_origin[1]), color, cv2.FILLED)
        cv2.putText(image, label, (label_origin[0], label_origin[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image

def get_colors_for_classes1(num_classes):
    """Generate a color for each class."""
    np.random.seed(42)  # For reproducible results
    return np.random.randint(0, 255, size=(num_classes, 3), dtype=int)

# Example usage
# Ensure the image is a uint8 array with values in [0, 255]
# frame = cv2.imread('example.jpg')  # Replace with actual image capture
# boxes = [(100, 150, 200, 250)]  # Example bounding boxes
# box_classes = [0]  # Example class indices
# class_names = ['object']
# scores = [0.85]  # Example scores

# output_image = draw_boxes_realtime(frame, boxes, box_classes, class_names, scores)
# cv2.imshow('Detection Capture', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
