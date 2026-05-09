from __future__ import print_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import warnings

from F3_DATASET import satellitedata
from F6_CROSSVAL import CrossVal
from F8_IMAGES4 import get_images4

# Import your CMX wrapper class
from CMX import CMX 

warnings.filterwarnings("ignore")


# =========================================================
# USER SETTINGS
# =========================================================
modelName = "FinaliremmodelLoRA.pt"
modelType = "CMX"
foldNo = 2
inputType = "all20Ch"

# Set your pretrained CMX checkpoint folder here
modelPath = r"C:\Users\Public\Server\experiments\LORA_MULTIMODAL_OLDS\Potsdam\Others_RandomModalityDropout\2026_1_27_17_6_model0"
CHECKPOINT_PATH = os.path.join(modelPath, modelName)
OUTPUT_JSON = os.path.join(modelPath, "potsdam_scenario_probabilities_cmx.json")





# Shared-only settings
alpha = 1.0
tau = 0.5
gamma_floor = 0.5
lam = 1e-3
rbf_sigma = 1.0

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev)

MASK_ARRAY = torch.tensor([
    [1, 0, 0],  # RGB only
    [0, 1, 0],  # NIR only
    [0, 0, 1],  # SWIR only
    [1, 1, 0],  # RGB + NIR
    [1, 0, 1],  # RGB + SWIR
    [0, 1, 1],  # NIR + SWIR
    [1, 1, 1],  # RGB + NIR + SWIR
], dtype=torch.float32)

SCENARIO_NAMES = [
    "RGB",
    "NIR",
    "SWIR",
    "RGB+NIR",
    "RGB+SWIR",
    "NIR+SWIR",
    "RGB+NIR+SWIR"
]


# =========================================================
# FORWARD HOOK COLLECTOR
# =========================================================
class LatentHookCollector:
    """
    Shared-only hook collector for CMX.

    We hook the last FFM output:
        model.core.backbone.ffm[3]

    This is the deepest fused/shared feature map before decoding.
    """
    def __init__(self, model):
        self.cache = {}
        self.handles = []

        self.handles.append(
            model.core.backbone.ffm[3].register_forward_hook(
                self._make_hook("z_shared")
            )
        )

    def _make_hook(self, name):
        def hook(module, inp, out):
            self.cache[name] = out.detach()
        return hook

    def clear(self):
        self.cache = {}

    def close(self):
        for h in self.handles:
            h.remove()


# =========================================================
# KERNEL
# =========================================================
def rbf_kernel(X, sigma=1.0):
    X = np.asarray(X, dtype=np.float64)
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    dist2 = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    K = np.exp(-dist2 / (2.0 * sigma ** 2))
    return K


# =========================================================
# DATA LOADER
# =========================================================
def build_training_generator():
    tsind, trind, vlind = CrossVal(5985, foldNo, 5)

    input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images4(
        5985, foldNo, 5, tsind, trind, vlind, inputType
    )

    params = {'batch_size': 1, 'shuffle': False}

    training_set = satellitedata(input_images[trind], target_masks[trind])
    training_generator = DataLoader(training_set, **params)

    meta_info = {
        "foldNo": foldNo,
        "inputType": inputType,
        "num_training_samples": len(trind),
        "num_validation_samples": len(vlind),
        "num_test_samples": len(tsind),
        "batch_size": params["batch_size"]
    }
    return training_generator, meta_info


# =========================================================
# CORE COMPUTATION (SHARED ONLY)
# =========================================================
def compute_scenario_statistics(model, data_loader):
    model.eval()
    hooks = LatentHookCollector(model)

    K = MASK_ARRAY.shape[0]
    d_sh_sum = torch.zeros(K, device=device)
    count_sum = torch.zeros(K, device=device)

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            B = images.size(0)

            # Full-modality forward
            hooks.clear()
            full_mask = torch.ones(B, 3, device=device)
            _ = model(images, mask=full_mask)

            if "z_shared" not in hooks.cache:
                raise RuntimeError(
                    "Shared hook did not capture any tensor. "
                    "Check whether model.core.backbone.ffm[3] is the correct hook point."
                )

            z_shared_full = hooks.cache["z_shared"]

            # Each scenario
            for k in range(K):
                scenario = MASK_ARRAY[k].to(device).unsqueeze(0).repeat(B, 1)

                hooks.clear()
                _ = model(images, mask=scenario)

                if "z_shared" not in hooks.cache:
                    raise RuntimeError(
                        f"Shared hook failed for scenario index {k} ({SCENARIO_NAMES[k]})."
                    )

                z_shared_s = hooks.cache["z_shared"]

                # Shared distortion only
                d_sh = F.mse_loss(
                    z_shared_s, z_shared_full, reduction='none'
                ).flatten(1).mean(dim=1)

                d_sh_sum[k] += d_sh.sum()
                count_sum[k] += B

    hooks.close()

    D_sh = (d_sh_sum / count_sum.clamp_min(1.0)).cpu().numpy()

    # Shared-only utility
    U = alpha * D_sh

    # 1D scenario descriptor
    X = D_sh[:, None]

    return D_sh, U, X


def compute_kernelized_scores(U, X, lam=1e-3, sigma=1.0):
    Kmat = rbf_kernel(X, sigma=sigma)
    I = np.eye(Kmat.shape[0], dtype=np.float64)

    # r = K (K + λI)^(-1) u
    inv_term = np.linalg.inv(Kmat + lam * I)
    r = Kmat @ inv_term @ U.reshape(-1, 1)
    r = r.squeeze(1)

    return Kmat, r


def scores_to_probabilities(r, tau=0.5, gamma_floor=0.5):
    r = np.asarray(r, dtype=np.float64)

    r_mean = np.mean(r)
    r_std = np.std(r)

    if r_std < 1e-12:
        r_norm = np.zeros_like(r)
    else:
        r_norm = (r - r_mean) / r_std

    exps = np.exp(tau * (r_norm - np.max(r_norm)))
    p_softmax = exps / exps.sum()

    K = len(p_softmax)
    uniform = np.ones(K, dtype=np.float64) / K
    p_final = (1.0 - gamma_floor) * uniform + gamma_floor * p_softmax
    p_final = p_final / p_final.sum()

    return p_softmax, p_final, r_norm


# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found:\n{CHECKPOINT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    print("Building training loader...")
    training_generator, meta_info = build_training_generator()

    print("Building CMX model...")
    model = CMX(num_classes=1, decoder_embed_dim=256, random_modality_dropout=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    print("Computing CMX shared-only scenario statistics from last FFM output...")
    D_sh, U, X = compute_scenario_statistics(model, training_generator)

    print("Computing kernelized scores...")
    Kmat, r = compute_kernelized_scores(U, X, lam=lam, sigma=rbf_sigma)
    p_softmax, p_final, r_norm = scores_to_probabilities(r, tau=tau, gamma_floor=gamma_floor)

    result = {
        "modelType": modelType,
        "checkpoint_path": CHECKPOINT_PATH,
        "output_json": OUTPUT_JSON,
        "hook_point": "model.core.backbone.ffm[3]",
        "alpha": alpha,
        "tau": tau,
        "gamma_floor": gamma_floor,
        "lambda": lam,
        "rbf_sigma": rbf_sigma,
        "scenario_names": SCENARIO_NAMES,
        "mask_array": MASK_ARRAY.tolist(),
        "meta_info": meta_info,
        "D_sh": D_sh.tolist(),
        "U": U.tolist(),
        "scenario_features_X": X.tolist(),
        "kernel_matrix": Kmat.tolist(),
        "kernel_scores_r": r.tolist(),
        "normalized_kernel_scores": r_norm.tolist(),
        "softmax_probs": p_softmax.tolist(),
        "final_probs": p_final.tolist()
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nScenario statistics:")
    for i, name in enumerate(SCENARIO_NAMES):
        print(
            f"{name:15s} | "
            f"D_sh={D_sh[i]:.6f} | "
            f"U={U[i]:.6f} | "
            f"r={r[i]:.6f} | "
            f"r_norm={r_norm[i]:.6f} | "
            f"p={p_final[i]:.6f}"
        )

    print(f"\nSaved probabilities to:\n{OUTPUT_JSON}")


if __name__ == "__main__":
    main()
