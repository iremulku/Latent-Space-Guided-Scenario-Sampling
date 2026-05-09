import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # opsiyonel ama genelde iyi oluyor
import torch 
import numpy as np
from F3_DATASET import satellitedata
from torch.utils.data import DataLoader
from F6_CROSSVAL import CrossVal
from F9_UNET_V2_3 import UNetV2 
from mmvit4_MissingGated import MMVit4
from mmvit4_Missing import MMVit5
from mmvit4 import MMVit4original
from RobustSeg import RobustMseg
from CMTFNet import CMTFNet
from RFNet import RFNet
from mmformer import mmformer
from MultiSenseSeg import Build_MultiSenseSeg
from CFFormer import CFFormer
from CMX import CMX 
from CMNeXt import CMNeXt
from FTransUNet import FTransUNet
from M3L import M3L
from RobSense import RobSense
from MetaRS import MetaRS
from MMANet import MMANet
from DFormer import DFormer
from base_vit import ViT
from seg_vit import SegWrapForViT
from F8_IMAGES4 import get_images4
from lora import LoRA_ViT
from lora_multimodal4 import LoRA_MMVit4
from lora_multimodal4CMX import LoRA_CMX
from F5_JACCARD2 import Jaccard2, JaccardAndF1

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


MODALITY_SETTING = 'swir_missing'   # 'nir_missing', 'rgb_only', vs. de yapabilirsin

def build_mod_mask(images, setting: str):
    B = images.size(0)
    mask = torch.ones(B, 3, device=images.device)

    if setting == 'full':
        return mask

    elif setting == 'rgb_missing':
        mask[:, 0] = 0
    elif setting == 'nir_missing':
        mask[:, 1] = 0
    elif setting == 'swir_missing':
        mask[:, 2] = 0
    elif setting == 'rgb_only':
        mask[:, 1] = 0
        mask[:, 2] = 0
    elif setting == 'nir_only':
        mask[:, 0] = 0
        mask[:, 2] = 0
    elif setting == 'swir_only':
        mask[:, 0] = 0
        mask[:, 1] = 0
    else:
        raise ValueError(f"Unknown MODALITY_SETTING: {setting}")

    return mask


# =========================================================
#   MODEL ve KAYITlı AĞI YÜKLE
# =========================================================
modelName = "FinaliremmodelLoRA.pt"   # eğitilmiş model dosyan
modelType = 'MMVit4'                 # veya 'LoRA_MMVit4', 'UNetV2', vs.
foldNo = 2
inputType = "all20Ch"              

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev)

if modelType == 'UNetV2':
    model = UNetV2(classes=1).to(device)
elif modelType == 'CMTFNet':
    model = CMTFNet(num_classes=1).to(device)
elif modelType == 'MMVit4':
    model = MMVit4().to(device)
elif modelType == 'MMVit5':
    model = MMVit5().to(device)   
elif modelType=='MMANet':              
    model = MMANet(height=224, width=224, num_classes=1)
elif modelType=='mmformer':              
    model = mmformer().to(device)  
elif modelType=='RFNet':              
    model = RFNet().to(device)    
elif modelType=='MultiSenseSeg':              
    model = Build_MultiSenseSeg(
        n_classes=1,
        in_chans=(3,3,3),
        aux=False
    ).to(device)     
elif modelType=='CMX':              
    model = CMX(num_classes=1, decoder_embed_dim=256, random_modality_dropout=False).to(device)         
elif modelType=='CMNeXt':              
    model = CMNeXt(backbone_variant="B2", decoder_embed_dim=256, random_modality_dropout=True).to(device) 
elif modelType=='FTransUNet':              
    model = FTransUNet(img_size=224, random_modality_dropout=True).to(device)
elif modelType=='MMVit4original':              
    model = MMVit4original().to(device)  
elif modelType=='CFFormer':              
    model = CFFormer( decoder_channels=(64, 128, 256, 512), decoder_embed_dim=256,num_classes=1, random_modality_dropout=True).to(device)                
elif modelType == 'RobustMseg':
    model = RobustMseg().to(device)
elif modelType=='RobSense':              
    model = RobSense(model_size="small", patch=16, num_classes=1).to(device)       
elif modelType=='M3L':              
    model = M3L(base=32, out_ch=1).to(device)   
elif modelType=='MetaRS':              
    model = MetaRS(backbone="resnet50", pretrained=True, num_classes=1).to(device) 
elif modelType=='DFormer':  
    model = DFormer(variant="S", num_classes=1)      
elif modelType == 'SegWrapForViT1':
    model1 = ViT('B_16_imagenet1k')
    lora_model = LoRA_ViT(model1, r=4).to(device)
    model = SegWrapForViT(vit_model=lora_model, image_size=224,
                          patches=16, dim=768, n_classes=1).to(device)
elif modelType == 'SegWrapForViT2':
    model1 = ViT('B_16_imagenet1k')
    model = SegWrapForViT(vit_model=model1, image_size=224,
                          patches=16, dim=768, n_classes=1).to(device)
elif modelType == 'SegWrapForViT':
    model1 = ViT('L_16_imagenet1k')
    lora_model = LoRA_ViT(model1, r=4).to(device)
    model = SegWrapForViT(vit_model=lora_model, image_size=224,
                          patches=16, dim=1024, n_classes=1).to(device)
elif modelType == 'LoRA_MMVit4':
    base_model = MMVit4().to(device)
    model = LoRA_MMVit4(base_model, r=4).to(device)
    
elif modelType =='LoRA_CMX':
    base_model = CMX(num_classes=1, decoder_embed_dim=256, random_modality_dropout=False)            
    model = LoRA_CMX(base_model,r=4,apply_linear=True,apply_conv2d=True,apply_decoder=False,verbose=True).to(device)                         
else:
    raise ValueError(f"Unknown modelType: {modelType}")

# Eğitilmiş modelin bulunduğu klasör:
modelPath = r"C:/Users/Public/Server/experiments/2026_4_19_18_3_model4"
#modelPath = r"C:/Users/Public/Server/experiments/LORA_MULTIMODAL_OLDS/DSTL/Latentfactorizaton/2025_12_23_11_27_model0"

# Ağı yükle
model.load_state_dict(torch.load(os.path.join(modelPath, modelName)))
model.eval()

# load input (for DSTL and RIT18) 

tsind,trind,vlind = CrossVal(5985,foldNo,5);
input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images4(5985, foldNo, 5, tsind, trind, vlind, inputType)


params = {'batch_size': 1, 'shuffle': False}    
test_set = satellitedata(input_images[tsind], target_masks[tsind])
test_generator = DataLoader(test_set, **params)

f1All = np.empty(test_generator.dataset.images.shape[0],dtype='float')
jcrdsAll = np.empty(test_generator.dataset.images.shape[0],dtype='float')

with torch.no_grad():
    ts = 0;
    for testim, testmas in test_generator:
        # the model
        images=testim.to(device)
        masks=testmas.to(device)
        

        if modelType in ['MMVit4', 'MMVit5', 'LoRA_CMX', 'LoRA_MMVit4', 'RFNet', "mmformer", "MultiSenseSeg", "CFFormer", "CMX", 'CMNeXt', 'FTransUNet', "M3L", 'RobSense', 'MMANet', 'DFormer']:
            mod_mask = build_mod_mask(images, MODALITY_SETTING)
            outputs = model(images, mask=mod_mask)
        else:
            outputs = model(images)


        masks = masks[:, 0, ...]   # Remove extra channel
        outputs = outputs[:, 0, ...]                     

        f1 = JaccardAndF1(torch.reshape(masks,(224*224,1)),torch.reshape(outputs,(224*224,1)))                                    
        jcrd = Jaccard2(torch.reshape(masks,(224*224,1)),torch.reshape(outputs,(224*224,1)))
        jcrdsAll[ts] = jcrd.to('cpu').numpy()[0]
        f1All[ts] = f1.to('cpu').numpy()[0]
        
       
        ts = ts+1;      

print(modelType + ", " + inputType + ", f1: ", f1All.mean() , "±" , f1All.std())
print(modelType + ", " + inputType + ", Jaccard: ", jcrdsAll.mean() , "±" , jcrdsAll.std())
            

