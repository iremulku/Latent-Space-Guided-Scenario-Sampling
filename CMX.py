import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 0) Minimal logger replacement (so no engine.logger dependency)
# ============================================================

class _Logger:
    def info(self, msg): print(msg)
    def warning(self, msg): print("[WARN]", msg)

def get_logger():
    return _Logger()

logger = get_logger()


# ============================================================
# 1) Utilities (kept simple; no distributed required)
# ============================================================

def load_state_dict(module, state_dict, strict=False, logger=None):
    """Lightweight state_dict loader (kept compatible with your original style)."""
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(m, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        m._load_from_state_dict(
            state_dict, prefix, local_metadata, True,
            all_missing_keys, unexpected_keys, err_msg
        )
        for name, child in m._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    missing_keys = [k for k in all_missing_keys if "num_batches_tracked" not in k]

    if unexpected_keys:
        err_msg.append("unexpected key(s): " + ", ".join(unexpected_keys))
    if missing_keys:
        err_msg.append("missing key(s): " + ", ".join(missing_keys))

    if err_msg:
        msg = "The model and loaded state dict do not match exactly:\n" + "\n".join(err_msg)
        if strict:
            raise RuntimeError(msg)
        else:
            print(msg)


def load_pretrain(model, filename, strict=False, revise_keys=[(r"^module\.", "")]):
    checkpoint = torch.load(filename, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    load_state_dict(model, state_dict, strict=strict)
    return checkpoint


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    """Initialize conv/norm layers in a module (or list of modules)."""
    def _init_one(m):
        for _, mm in m.named_modules():
            if isinstance(mm, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                conv_init(mm.weight, **kwargs)
                if mm.bias is not None:
                    nn.init.constant_(mm.bias, 0)
            elif isinstance(mm, norm_layer) or isinstance(mm, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                if hasattr(mm, "eps"):
                    mm.eps = bn_eps
                if hasattr(mm, "momentum"):
                    mm.momentum = bn_momentum
                if mm.weight is not None:
                    nn.init.constant_(mm.weight, 1)
                if mm.bias is not None:
                    nn.init.constant_(mm.bias, 0)

    if isinstance(module_list, list):
        for m in module_list:
            _init_one(m)
    else:
        _init_one(module_list)


# ============================================================
# 2) Feature Rectify + Fusion blocks (from your code, kept)
# ============================================================

def trunc_normal_(tensor, mean=0., std=1.):
    # simple replacement for timm trunc_normal_
    with torch.no_grad():
        return tensor.normal_(mean, std)

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        avg = self.avg_pool(x).view(B, self.dim * 2)
        mx  = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, mx), dim=1)  # B,4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2,B,C,1,1
        return channel_weights

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2,B,1,H,W
        return spatial_weights

class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        cw = self.channel_weights(x1, x2)
        sw = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * cw[1] * x2 + self.lambda_s * sw[1] * x2
        out_x2 = x2 + self.lambda_c * cw[0] * x1 + self.lambda_s * sw[0] * x1
        return out_x1, out_x2

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1_out = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2_out = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        return x1_out, x2_out

class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)  # B,HW,C
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)         # B,HW,C
        merge = torch.cat((x1, x2), dim=-1) # B,HW,2C
        merge = self.channel_emb(merge, H, W)  # B,C,H,W
        return merge


# ============================================================
# 3) A SELF-CONTAINED Dual Backbone (no external encoders)
#    - two streams: rgb and modal_x (both 3ch)
#    - outputs 4 multi-scale fused features for decoder
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DualSimpleBackbone(nn.Module):
    """
    Dual stream CNN backbone + FRM + FFM at each stage.
    Output: list of 4 fused feature maps with channels [64, 128, 320, 512]
    """
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.channels = [64, 128, 320, 512]

        # Stream stems
        self.rgb_stem   = ConvBlock(3, 64, stride=2, norm_layer=norm_layer)    # H/2
        self.mod_stem   = ConvBlock(3, 64, stride=2, norm_layer=norm_layer)    # H/2

        # Stage downsample blocks (each stage halves resolution)
        self.rgb_s1 = ConvBlock(64, 64,  stride=2, norm_layer=norm_layer)      # H/4
        self.mod_s1 = ConvBlock(64, 64,  stride=2, norm_layer=norm_layer)

        self.rgb_s2 = ConvBlock(64, 128, stride=2, norm_layer=norm_layer)      # H/8
        self.mod_s2 = ConvBlock(64, 128, stride=2, norm_layer=norm_layer)

        self.rgb_s3 = ConvBlock(128, 320, stride=2, norm_layer=norm_layer)     # H/16
        self.mod_s3 = ConvBlock(128, 320, stride=2, norm_layer=norm_layer)

        self.rgb_s4 = ConvBlock(320, 512, stride=2, norm_layer=norm_layer)     # H/32
        self.mod_s4 = ConvBlock(320, 512, stride=2, norm_layer=norm_layer)

        # FRM + FFM per stage
        self.frm = nn.ModuleList([
            FeatureRectifyModule(64,  reduction=1),
            FeatureRectifyModule(128, reduction=1),
            FeatureRectifyModule(320, reduction=1),
            FeatureRectifyModule(512, reduction=1),
        ])
        # heads are modest; must divide dim
        self.ffm = nn.ModuleList([
            FeatureFusionModule(64,  reduction=1, num_heads=8, norm_layer=norm_layer),
            FeatureFusionModule(128, reduction=1, num_heads=8, norm_layer=norm_layer),
            FeatureFusionModule(320, reduction=1, num_heads=8, norm_layer=norm_layer),
            FeatureFusionModule(512, reduction=1, num_heads=8, norm_layer=norm_layer),
        ])

    def forward(self, rgb, modal_x):
        outs = []

        # Stage 0 -> produce 64 @ H/4 (stem H/2 then stage1 H/4)
        r = self.rgb_stem(rgb)
        m = self.mod_stem(modal_x)
        r = self.rgb_s1(r)
        m = self.mod_s1(m)
        r, m = self.frm[0](r, m)
        f = self.ffm[0](r, m)
        outs.append(f)  # 64, H/4

        # Stage 1 -> 128 @ H/8
        r = self.rgb_s2(r)
        m = self.mod_s2(m)
        r, m = self.frm[1](r, m)
        f = self.ffm[1](r, m)
        outs.append(f)

        # Stage 2 -> 320 @ H/16
        r = self.rgb_s3(r)
        m = self.mod_s3(m)
        r, m = self.frm[2](r, m)
        f = self.ffm[2](r, m)
        outs.append(f)

        # Stage 3 -> 512 @ H/32
        r = self.rgb_s4(r)
        m = self.mod_s4(m)
        r, m = self.frm[3](r, m)
        f = self.ffm[3](r, m)
        outs.append(f)

        return outs  # [c1,c2,c3,c4]


# ============================================================
# 4) Decoder (MLP-like) - self-contained
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: B,C,H,W -> B,HW,C -> B,HW,embed
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class DecoderHead(nn.Module):
    def __init__(self, in_channels=(64, 128, 320, 512), num_classes=1, dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d, embed_dim=256, align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

        c1, c2, c3, c4 = in_channels
        self.linear_c4 = MLP(c4, embed_dim)
        self.linear_c3 = MLP(c3, embed_dim)
        self.linear_c2 = MLP(c2, embed_dim)
        self.linear_c1 = MLP(c1, embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        if self.dropout is not None:
            _c = self.dropout(_c)
        out = self.linear_pred(_c)  # B,num_classes,H/4,W/4
        return out


# ============================================================
# 5) EncoderDecoder (self-contained version for your DSTL setup)
# ============================================================

class SimpleCfg:
    def __init__(self,
                 num_classes=1,
                 decoder_embed_dim=256,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 pretrained_model=None):
        self.num_classes = num_classes
        self.decoder_embed_dim = decoder_embed_dim
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.pretrained_model = pretrained_model


class EncoderDecoder(nn.Module):
    """
    Dual-input encoder-decoder.
    This version is self-contained (no .encoders, no timm, no engine imports).
    """
    def __init__(self, cfg=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cfg = cfg if cfg is not None else SimpleCfg()
        self.norm_layer = norm_layer

        # Backbone (dual stream)
        self.backbone = DualSimpleBackbone(norm_layer=norm_layer)
        self.channels = self.backbone.channels

        # Decoder
        self.decode_head = DecoderHead(
            in_channels=self.channels,
            num_classes=self.cfg.num_classes,
            norm_layer=norm_layer,
            embed_dim=self.cfg.decoder_embed_dim
        )

        # init decoder weights a bit like your original style
        logger.info("Initializing weights ...")
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    norm_layer, self.cfg.bn_eps, self.cfg.bn_momentum,
                    mode="fan_in", nonlinearity="relu")

        # Optional pretrained load (if you want to load something you saved)
        if self.cfg.pretrained_model:
            logger.info(f"Loading pretrained model: {self.cfg.pretrained_model}")
            load_pretrain(self, self.cfg.pretrained_model, strict=False)

    def encode_decode(self, rgb, modal_x):
        """
        rgb:     [B,3,H,W]
        modal_x: [B,3,H,W]
        returns logits [B,num_classes,H,W]
        """
        ori_h, ori_w = rgb.shape[2], rgb.shape[3]
        feats = self.backbone(rgb, modal_x)      # 4-scale list
        out = self.decode_head(feats)            # B,C,H/4,W/4
        out = F.interpolate(out, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
        return out

    def forward(self, rgb, modal_x):
        return self.encode_decode(rgb, modal_x)


# ============================================================
# 6) DSTL wrapper: accepts x=[B,3,3,H,W], target=[B,3,1,H,W]
#    + Random modality dropout (non-empty 7 masks)
# ============================================================

MASK_ARRAY_3MOD = torch.tensor([
    [1, 0, 0],  # RGB only
    [0, 1, 0],  # NIR only
    [0, 0, 1],  # SWIR only
    [1, 1, 0],  # RGB + NIR
    [1, 0, 1],  # RGB + SWIR
    [0, 1, 1],  # NIR + SWIR
    [1, 1, 1],  # RGB + NIR + SWIR
], dtype=torch.float32)


class sigmoidOut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(sigmoidOut, self).__init__()
        self.outsig= nn.Sigmoid()

    def forward(self, x):
        x = self.outsig(x)
        return x  


class CMX(nn.Module):
    """
    Input:
        x:      [B,3,3,H,W]  (RGB,NIR,SWIR each 3-ch)
        target: [B,3,1,H,W]  (binary mask repeated across modalities)
    Output:
        logits: [B,3,1,H,W]  (raw logits; use BCEWithLogitsLoss)
    """
    def __init__(self,
                 num_classes=1,
                 decoder_embed_dim=256,
                 random_modality_dropout=True):
        super().__init__()
        assert num_classes == 1, "This file is configured for binary segmentation (num_classes=1)."

        cfg = SimpleCfg(num_classes=num_classes, decoder_embed_dim=decoder_embed_dim)
        self.core = EncoderDecoder(cfg=cfg, norm_layer=nn.BatchNorm2d)

        # Build modal_x from (NIR, SWIR) => 6 channels -> 3 channels
        self.modal_proj = nn.Conv2d(6, 3, kernel_size=1, bias=False)

        self.random_modality_dropout = bool(random_modality_dropout)
        self.register_buffer("mask_array", MASK_ARRAY_3MOD, persistent=False)
        self.outsig = sigmoidOut(1,1)

    @torch.no_grad()
    def _sample_mask(self, B, device, dtype):
        idx = torch.randint(0, self.mask_array.shape[0], (B,), device=device)
        return self.mask_array[idx].to(device=device, dtype=dtype)  # [B,3]

    def forward(self, x, target=None, mask=None, return_loss=False):
        """
        x: [B,3,3,H,W]
        target: [B,3,1,H,W] float 0/1 (optional)
        mask: [B,3] optional
        return_loss: if True and target provided -> returns (loss, logits)
        """
        if x.dim() != 5 or x.size(1) != 3 or x.size(2) != 3:
            raise ValueError(f"Expected x=[B,3,3,H,W], got {tuple(x.shape)}")

        B, _, _, H, W = x.shape

        # Decide modality mask
        if mask is None:
            if self.training and self.random_modality_dropout:
                mask = self._sample_mask(B, x.device, x.dtype)
            else:
                mask = torch.ones((B, 3), device=x.device, dtype=x.dtype)
        else:
            mask = mask.to(device=x.device, dtype=x.dtype)
            if mask.shape != (B, 3):
                raise ValueError(f"mask must be [B,3], got {tuple(mask.shape)}")

        # Split modalities
        rgb  = x[:, 0]  # [B,3,H,W]
        nir  = x[:, 1]
        swir = x[:, 2]

        # Apply dropout (zero missing)
        m0 = mask[:, 0].view(B, 1, 1, 1)
        m1 = mask[:, 1].view(B, 1, 1, 1)
        m2 = mask[:, 2].view(B, 1, 1, 1)
        rgb  = rgb  * m0
        nir  = nir  * m1
        swir = swir * m2

        # Create 2nd stream input (modal_x) from nir+swir
        modal_x = self.modal_proj(torch.cat([nir, swir], dim=1))  # [B,3,H,W]

        # Core model logits: [B,1,H,W]
        logits_1 = self.core(rgb, modal_x)  # raw logits

        # Match your target format: [B,3,1,H,W]
        logits = logits_1.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        if return_loss and (target is not None):
            target = target.to(device=logits.device, dtype=logits.dtype)
            if target.shape != logits.shape:
                raise ValueError(f"target shape {tuple(target.shape)} != logits shape {tuple(logits.shape)}")

            loss = F.binary_cross_entropy_with_logits(logits, target)
            return loss, logits
        
        logits = self.outsig(logits)
        return logits

