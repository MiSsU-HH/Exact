# from models.Exact_cls import Exact
from models.TSViT_seg import TSViT

def get_model(config, device):
    model_config = config['MODEL']

    # if model_config['architecture'] == "Exact":
    #     model_config['device'] = device
    #     return Exact(model_config).to(device)
    
    if model_config['architecture'] == "TSViT":
        return TSViT(model_config).to(device)
    
    else:
        raise NameError("Model architecture %s not found, choose from: 'Exact', 'TSViT'")
