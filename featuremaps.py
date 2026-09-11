
import torch
from torchvision import transforms

def get_module_by_name(model, name):
    module = model
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module

def visualize_feature_maps(model, input_image, target):
    activations = {}
    def hook_fn(module, input, output):
        activations['feat'] = output.detach()
    if isinstance(target, str):
        target_module = get_module_by_name(model, target)
    else:
        target_module = target
    hook = target_module.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(input_image)
    hook.remove()
    feat = activations['feat'].squeeze().mean(dim=0)
    return feat
