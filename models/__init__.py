from models.Exact.Exact_cls import Exact_cls
from models.Exact.TSViT_seg import TSViT_seg

def get_model(config, device):
    model_config = config['MODEL']

    if model_config['architecture'] == "Exact_cls":
        model_config['device'] = device
        return Exact_cls(model_config).to(device)
    
    if model_config['architecture'] == "TSViT_seg":
        return TSViT_seg(model_config).to(device)
    
    else:
        raise NameError("Model architecture %s not found, choose from: 'Exact_cls', 'TSViT_seg'")
