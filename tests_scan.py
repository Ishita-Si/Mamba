import torch
from src.model import selective_scan_seq, selective_scan_par, MambaBlock
torch.manual_seed(0)
b,l,d,n=2,37,8,4
u=torch.randn(b,l,d); delta=torch.rand(b,l,d)*0.5; A=-torch.rand(d,n)*5; B=torch.randn(b,l,n); C=torch.randn(b,l,n); D=torch.randn(d)
y1=selective_scan_seq(u,delta,A,B,C,D); y2=selective_scan_par(u,delta,A,B,C,D)
print("scan max abs diff:", (y1-y2).abs().max().item()); assert torch.allclose(y1,y2,atol=1e-5)
blk=MambaBlock(16)
x=torch.randn(1,20,16); x2=x.clone(); x2[:,10:]+=torch.randn(1,10,16)
assert torch.allclose(blk(x)[:,:10],blk(x2)[:,:10],atol=1e-5); print("causality OK")
