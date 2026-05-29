import numpy as np, struct, os

W = open(r"D:\caal2026\model_params\weights.bin","rb").read()
def f32(off,n): return np.frombuffer(W[off:off+4*n], dtype="<f4").astype(np.float64).copy()
hidx = f32(0,4096).astype(np.int64)
projW=f32(16384,64); projB=f32(16640,64)
def s4dp(b): return (f32(b,64),f32(b+256,2048).reshape(64,32),f32(b+256+8192,2048).reshape(64,32),
                      f32(b+256+8192+8192,4096).reshape(64,32,2),f32(b+256+8192+8192+16384,64))
s1=s4dp(16896); s2=s4dp(50176); fcW=f32(83456,256).reshape(4,64); fcB=f32(84480,4)
def s4d(u,p):
    logdt,logar,aimag,C,D=p; T,H=u.shape; out=np.zeros_like(u)
    for h in range(H):
        dt=np.exp(logdt[h]); Ar=-np.exp(logar[h]); Ai=aimag[h]
        Abar=np.exp(Ar*dt)*(np.cos(Ai*dt)+1j*np.sin(Ai*dt)); A=Ar+1j*Ai; Cc=C[h,:,0]+1j*C[h,:,1]
        Ct=Cc*(Abar-1)/A; t=np.arange(T); K=2*np.real((Ct[None,:]*(Abar[None,:]**t[:,None])).sum(1))
        out[:,h]=D[h]*u[:,h]+np.convolve(K,u[:,h])[:T]
    return out
gelu=lambda x:0.5*x*(1+np.tanh(0.79788456*(x+0.044715*x**3)))
true=[3,3,3,2,2,2,1,1,1,0,0,0]
for n in range(12):
    img=np.frombuffer(open(rf"D:\caal2026\test_data\sample_{n:02d}_input.bin","rb").read(),dtype="<f4").astype(np.float64)
    proj=img[hidx][:,None]*projW[None,:]+projB[None,:]
    g2=gelu(s4d(gelu(s4d(proj,s1)),s2)); pooled=g2[4095]; logits=fcW@pooled+fcB
    p=np.exp(logits-logits.max()); p/=p.sum(); pred=int(np.argmax(p))
    print(f"sample_{n:02d} pred={pred} true={true[n]} {'OK ' if pred==true[n] else 'MISS'} probs=[{', '.join('%.4f'%x for x in p)}]")
