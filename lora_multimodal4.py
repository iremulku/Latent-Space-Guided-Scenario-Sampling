import torch
import torch.nn as nn
import math

class _LoRALayer(nn.Module):
    """Linear katmanlar için LoRA wrapper."""
    def __init__(self, base_layer: nn.Linear, r: int):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        self.reset_parameters()

        # Ana layer'ı dondur
        for p in self.base.parameters():
            p.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # W_eff(x) = W_base(x) + B(A(x))
        return self.base(x) + self.lora_B(self.lora_A(x))


class _LoRAConv3d(nn.Module):
    """Conv3d katmanı için LoRA benzeri düşük-rank güncelleme."""
    def __init__(self, base_conv: nn.Conv3d, r: int):
        super().__init__()
        self.base = base_conv
        self.r = r

        in_ch = base_conv.in_channels
        out_ch = base_conv.out_channels

        # 1x1x1 conv ile düşük-rank güncelleme
        self.lora_A = nn.Conv3d(in_ch, r, kernel_size=1, stride=1, padding=0, bias=False)
        self.lora_B = nn.Conv3d(r, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

        self.reset_parameters()

        # Ana conv ağırlıklarını dondur
        for p in self.base.parameters():
            p.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Conv_eff(x) = Conv_base(x) + B(A(x))
        return self.base(x) + self.lora_B(self.lora_A(x))

def LoRA_MMVit4(model, r=4):
    """
    MMVit4 modeli için:
    - Önce tüm parametreleri dondurur,
    - multimodal_transformer içindeki attention linear katmanlarına (qkv, proj) LoRA ekler,
    - EarlyFusionBlock içindeki 1x1x1 Conv3d (self.conv) katmanlarına da Conv-LoRA ekler.
    """

    # 1) Tüm mevcut parametreleri dondur
    for p in model.parameters():
        p.requires_grad = False

    # 2) multimodal_transformer var mı kontrol et
    if hasattr(model, "multimodal_transformer"):
        mm_trans = model.multimodal_transformer

        # Transformer içindeki attention bloklarını gez
        if hasattr(mm_trans, "cross_attention_list"):
            for blk in mm_trans.cross_attention_list:
                # blk: Residual
                pre = getattr(blk, "fn", None)     # PreNormDrop
                if pre is None:
                    continue

                attn = getattr(pre, "fn", None)    # SelfAttention olmalı
                if attn is None:
                    continue

                # qkv ve proj Linear ise LoRA uygula
                if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                    attn.qkv = _LoRALayer(attn.qkv, r=r)

                if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                    attn.proj = _LoRALayer(attn.proj, r=r)
        else:
            print("Uyarı: 'multimodal_transformer' içinde 'cross_attention_list' bulunamadı.")
    else:
        print("Uyarı: Modelde 'multimodal_transformer' bulunamadı.")

    # 3) EarlyFusionBlock'lara Conv-LoRA ekle
    # MMVit4 içinde: fusion1, fusion2, fusion3, fusion4, fusion5, fusion6
    fusion_names = ["fusion1", "fusion2", "fusion3", "fusion4", "fusion5", "fusion6"]
    for name in fusion_names:
        if hasattr(model, name):
            fusion_block = getattr(model, name)
            # EarlyFusionBlock: self.conv bir Conv3d
            if hasattr(fusion_block, "conv") and isinstance(fusion_block.conv, nn.Conv3d):
                fusion_block.conv = _LoRAConv3d(fusion_block.conv, r=r)

    return model

