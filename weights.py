is the formatting w# save_weights.py — run this in your M2 Python environment
import numpy as np
import torch
from your_m2_model import S4DModel   # adjust import to your actual file

model = S4DModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

def save_f32(arr, path):
    np.array(arr, dtype=np.float32).flatten().tofile(path)

# Export weights
save_f32(model.input_proj.weight.detach().numpy(), 'weights/linear_w.bin')
save_f32(model.input_proj.bias.detach().numpy(),   'weights/linear_b.bin')
save_f32(model.s4d1.A.real.detach().numpy(),       'weights/s4d1_A_real.bin')
save_f32(model.s4d1.A.imag.detach().numpy(),       'weights/s4d1_A_imag.bin')
save_f32(model.s4d1.B.detach().numpy(),            'weights/s4d1_B.bin')
save_f32(model.s4d1.C.real.detach().numpy(),       'weights/s4d1_C_real.bin')
save_f32(model.s4d1.C.imag.detach().numpy(),       'weights/s4d1_C_imag.bin')
save_f32(model.s4d2.A.real.detach().numpy(),       'weights/s4d2_A_real.bin')
save_f32(model.s4d2.A.imag.detach().numpy(),       'weights/s4d2_A_imag.bin')
save_f32(model.s4d2.B.detach().numpy(),            'weights/s4d2_B.bin')
save_f32(model.s4d2.C.real.detach().numpy(),       'weights/s4d2_C_real.bin')
save_f32(model.s4d2.C.imag.detach().numpy(),       'weights/s4d2_C_imag.bin')
save_f32(model.fc.weight.detach().numpy(),         'weights/fc_w.bin')
save_f32(model.fc.bias.detach().numpy(),           'weights/fc_b.bin')

# Export 10 test samples
import os; os.makedirs('testdata', exist_ok=True); os.makedirs('weights', exist_ok=True)
from torch.utils.data import DataLoader
from your_dataset import GalaxyDataset

dataset = GalaxyDataset('test')
loader  = DataLoader(dataset, batch_size=1, shuffle=False)

for i, (img, label) in enumerate(loader):
    if i >= 10: break
    with torch.no_grad():
        probs = model(img).softmax(-1).numpy().flatten()
    save_f32(img.numpy(), f'testdata/input_{i:02d}.bin')
    save_f32(probs,       f'testdata/ref_out_{i:02d}.bin')
    np.savetxt(f'testdata/label_{i:02d}.txt', [int(label)], fmt='%d')
    print(f'Sample {i}: label={int(label)}, probs={probs}')

print('Done. Weights in weights/, test data in testdata/')
rite for it?
