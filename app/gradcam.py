import torch
import cv2
import numpy as np
import os
from torchvision.models import resnet18

STATIC_PATH = 'app/static/gradcam_results'

def generate_gradcam(model, input_tensor, side):
    target_layer = model.layer4[-1]
    gradients = []
    activations = []

    def save_gradients_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activations_hook(module, input, output):
        activations.append(output)

    handle1 = target_layer.register_forward_hook(save_activations_hook)
    handle2 = target_layer.register_backward_hook(save_gradients_hook)

    output = model(input_tensor)
    pred_class = output.argmax().item()
    model.zero_grad()
    output[0, pred_class].backward()

    grads_val = gradients[0].detach().numpy()[0]
    acts_val = activations[0].detach().numpy()[0]

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(acts_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts_val[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    input_image = input_tensor.squeeze().permute(1, 2, 0).detach().numpy()
    input_image = np.clip((input_image * [0.229, 0.224, 0.225]) +
                          [0.485, 0.456, 0.406], 0, 1)
    input_image = np.uint8(255 * input_image)

    overlay = cv2.addWeighted(input_image, 0.5, cam, 0.5, 0)
    filename = f'{side}_gradcam.jpg'
    filepath = os.path.join(STATIC_PATH, filename)
    cv2.imwrite(filepath, overlay[:, :, ::-1])

    handle1.remove()
    handle2.remove()

    return f"/static/gradcam_results/{filename}"