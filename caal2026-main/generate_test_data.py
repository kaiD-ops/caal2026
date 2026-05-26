import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.gclassifier import GalaxyClassifierS4D

CKPT_PATH = "model_params/galaxys4-29070.pth"
checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)

C_IN = state_dict["uproject.weight"].shape[1]
is_colored = (C_IN == 3)
print(f"Model channels: C_IN={C_IN} ({'RGB' if is_colored else 'grayscale'})")

model = GalaxyClassifierS4D(colored=is_colored)
model.load_state_dict(state_dict)
model.eval()

WEIGHT_OUT = "model_weights.bin"
param_order = [
    ("hilbert_scan.indices", "int32"),
    ("uproject.weight",      "float32"),
    ("uproject.bias",        "float32"),
    ("s4_1.log_dt",          "float32"),
    ("s4_1.log_A_real",      "float32"),
    ("s4_1.A_imag",          "float32"),
    ("s4_1.C",               "float32"),
    ("s4_1.D",               "float32"),
    ("s4_2.log_dt",          "float32"),
    ("s4_2.log_A_real",      "float32"),
    ("s4_2.A_imag",          "float32"),
    ("s4_2.C",               "float32"),
    ("s4_2.D",               "float32"),
    ("fc.weight",            "float32"),
    ("fc.bias",              "float32"),
]

total_bytes = 0
with open(WEIGHT_OUT, "wb") as f:
    for key, dtype in param_order:
        arr = state_dict[key].numpy().astype(dtype)
        f.write(arr.tobytes())
        total_bytes += arr.nbytes
        print(f"  {key:35s} shape={str(arr.shape):20s} bytes={arr.nbytes}")
print(f"\nWrote {WEIGHT_OUT}: {total_bytes} bytes total\n")

intermediates = {}

def make_hook(name):
    def hook(module, inp, output):
        out = output[0] if isinstance(output, tuple) else output
        intermediates[name] = out.detach().cpu().numpy().astype(np.float32)
    return hook

hooks = [
    model.hilbert_scan.register_forward_hook(make_hook("hilbert")),
    model.uproject    .register_forward_hook(make_hook("linear")),
    model.s4_1        .register_forward_hook(make_hook("s4d1")),
    model.act1        .register_forward_hook(make_hook("gelu1")),
    model.s4_2        .register_forward_hook(make_hook("s4d2")),
    model.act2        .register_forward_hook(make_hook("gelu2")),
    model.take_last   .register_forward_hook(make_hook("pooled")),
    model.fc          .register_forward_hook(make_hook("logits")),
]

try:
    from galaxy_mnist import GalaxyMNIST
    import torchvision.transforms as T
    gray_tfm = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
    rgb_tfm  = T.Compose([T.ToTensor()])
    transform = rgb_tfm if is_colored else gray_tfm
    dataset = GalaxyMNIST(root="./data", train=False, download=True, transform=transform)
    from collections import defaultdict
    class_buckets = defaultdict(list)
    for img, lbl in dataset:
        l = int(lbl)
        if len(class_buckets[l]) < 3:
            class_buckets[l].append((img, l))
        if all(len(v) >= 3 for v in class_buckets.values()):
            break
    samples = [item for bucket in class_buckets.values() for item in bucket]
except Exception as e:
    print(f"[WARN] Could not load GalaxyMNIST ({e}). Using random tensors.")
    C = 3 if is_colored else 1
    samples = [(torch.rand(C, 64, 64), i % 4) for i in range(12)]

os.makedirs("test_data", exist_ok=True)
class_names = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Spiral"]

for i, (img, label) in enumerate(samples):
    intermediates.clear()
    img_batch = img.unsqueeze(0)
    with torch.no_grad():
        probs  = model(img_batch)
        logits = model(img_batch, return_logits=True)
    prefix = f"test_data/sample_{i:02d}"
    img.numpy().astype(np.float32).flatten().tofile(f"{prefix}_input.bin")
    with open(f"{prefix}_label.txt", "w") as f:
        f.write(str(int(label)))
    for name, arr in intermediates.items():
        arr.flatten().tofile(f"{prefix}_{name}.bin")
    probs.numpy().astype(np.float32).flatten().tofile(f"{prefix}_probs.bin")
    pred = int(probs.argmax())
    print(f"  sample {i:02d}: true={label}({class_names[int(label)]}), pred={pred}({class_names[pred]}), conf={float(probs[0,pred]):.3f}")

for h in hooks:
    h.remove()

with open("test_data/metadata.txt", "w") as f:
    f.write(f"num_samples={len(samples)}\nc_in={C_IN}\nseq_len=4096\nd_model=64\nd_state=32\nn_classes=4\n")

print(f"\nDone. Exported {len(samples)} samples to test_data/")
