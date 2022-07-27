import torch
import torch.nn as nn
import torch.nn.functional as F

# +
from .cswin_transformer import CSWinTransformer
from timm.models import create_model

def cswin_b(pretrained=True):
    model = create_model(
        'CSWin_96_24322_base_384',
        pretrained=True,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        img_size=384) 
    
    if pretrained == True:
        weight = torch.load('/home/u6859530/DFUC/pretrained/cswin_base_384.pth')
        model.load_state_dict(weight['state_dict_ema'], strict=False)

    model.head = nn.Identity()
    return model

# +
#from .cswin_transformer_v2 import CSWinTransformer_v2
#from timm.models import create_model
#
#def cswin_v2_b(pretrained=True):
#    model = create_model(
#        'CSWin_96_24322_base_384',
#        pretrained=True,
#        num_classes=1000,
#        drop_rate=0.0,
#        drop_path_rate=0.1,
#        img_size=384) 
#    
#    if pretrained == True:
#        weight = torch.load('/home/u4952464/DFUC/Former_segmenter/pretrained/cswin_base_384.pth')
#        model.load_state_dict(weight['state_dict_ema'], strict=False)
#
#    model.head = nn.Identity()
#    return model

# +
#from .cswin_transformer_v3 import CSWinTransformer_v3
#from timm.models import create_model
#
#def cswin_v3_b(pretrained=True):
#    model = create_model(
#        'CSWin_96_24322_base_384',
#        pretrained=True,
#        num_classes=1000,
#        drop_rate=0.0,
#        drop_path_rate=0.1,
#        img_size=384) 
#    
#    if pretrained == True:
#        weight = torch.load('/home/u4952464/DFUC/Former_segmenter/pretrained/cswin_base_384.pth')
#        model.load_state_dict(weight['state_dict_ema'], strict=False)
#
#    model.head = nn.Identity()
#    return model

# +
#from .cswin_transformer_v4 import CSWinTransformer_v4
#from timm.models import create_model
#
#def cswin_v4_b(pretrained=True):
#    model = create_model(
#        'CSWin_96_24322_base_384',
#        pretrained=True,
#        num_classes=1000,
#        drop_rate=0.0,
#        drop_path_rate=0.1,
#        img_size=384) 
#    
#    if pretrained == True:
#        weight = torch.load('/home/u4952464/DFUC/Former_segmenter/pretrained/cswin_base_384.pth')
#        model.load_state_dict(weight['state_dict_ema'], strict=False)
#
#    model.head = nn.Identity()
#    return model
