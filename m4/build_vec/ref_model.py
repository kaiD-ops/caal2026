import numpy as np, struct

W = open(r"D:\caal2026\model_params\weights.bin","rb").read()
img = np.frombuffer(open(r"D:\caal2026\test_data\sample_09_input.bin","rb").read(), dtype="<f4").astype(np.float64)

def f32(off,n): return np.frombuffer(W[off:off+4*n], dtype="<f4").astype(np.float64).copy()
def i32(off,n): return np.frombuffer(W[off:off+4*n], dtype="<i4").copy()

hidx = f32(0,4096).astype(np.int64)   # indices stored as float32
projW = f32(16384,64); projB = f32(16640,64)
def s4d_params(base):
    logdt=f32(base,64); logar=f32(base+256,2048).reshape(64,32)
    aimag=f32(base+256+8192,2048).reshape(64,32)
    C=f32(base+256+8192+8192,4096).reshape(64,32,2); D=f32(base+256+8192+8192+16384,64)
    return logdt,logar,aimag,C,D
s1=s4d_params(16896); s2=s4d_params(50176)
fcW=f32(83456,256).reshape(4,64); fcB=f32(84480,4)

# hilbert + projection (in_dim=1)
hil = img[hidx]                       # [4096]
proj = hil[:,None]*projW[None,:] + projB[None,:]   # [4096,64]

def s4d(u, params):
    logdt,logar,aimag,C,D = params
    T,H = u.shape; N=32
    out = np.zeros_like(u)
    for h in range(H):
        dt=np.exp(logdt[h])
        Ar=-np.exp(logar[h]); Ai=aimag[h]               # [32]
        Abar=np.exp(Ar*dt)*(np.cos(Ai*dt)+1j*np.sin(Ai*dt))
        A=Ar+1j*Ai; Cc=C[h,:,0]+1j*C[h,:,1]
        Ct=Cc*(Abar-1)/A
        t=np.arange(T)
        # K[t]=2*Re(sum_n Ct_n * Abar_n^t)
        K=2*np.real((Ct[None,:]*(Abar[None,:]**t[:,None])).sum(axis=1))  # [T]
        uh=u[:,h]
        # causal conv y[t]=D*u[t]+sum_{j=0..t}K[j]u[t-j]
        conv=np.convolve(K,uh)[:T]
        out[:,h]=D[h]*uh+conv
    return out

def gelu(x): return 0.5*x*(1+np.tanh(0.79788456*(x+0.044715*x**3)))

s4d1=s4d(proj,s1)
g1=gelu(s4d1)
s4d2=s4d(g1,s2)
g2=gelu(s4d2)
pooled=g2[4095]                        # [64]
logits=fcW@pooled+fcB
probs=np.exp(logits-logits.max()); probs/=probs.sum()

r=lambda a:[round(float(x),5) for x in a]
for t in (0,1,64,1024,4095):
    print(f"s4d1pre[{t},0:4]", r(s4d1[t,:4]))
print("s4d1 PRE-gelu[0,0:8]", r(s4d1[0,:8]))
print("s4d2 PRE-gelu[0,0:8]", r(s4d2[0,:8]))
print("buf_proj[0,0:4] ", r(proj[0,:4]))
print("buf_s4d1 g1[0,0:4]", r(g1[0,:4]))
print("buf_s4d2 g2[0,0:4]", r(g2[0,:4]))
print("buf_pooled[0:8] ", r(pooled[:8]))
print("logits          ", r(logits))
print("probs           ", r(probs), "argmax", int(np.argmax(probs)))
