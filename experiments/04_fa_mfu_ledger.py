# -*- coding: utf-8 -*-
"""FA 三代利用率对账 + 同题三路线账. 对应 docs/02."""
A100, H100, H100_fp8 = 312e12, 989e12, 1979e12   # bf16/fp16/fp8 峰值 TFLOPs
print("=== 三代 MFU 对账 ===")
for n, t, p in [("FA2@A100 bf16", 225e12, A100), ("FA3@H100 FP16", 740e12, H100),
                ("FA3@H100 FP8", 1192e12, H100_fp8)]:
    print("%-14s %5.0f TFLOPs / %5.0f 峰值 = %.0f%%" % (n, t/1e12, p/1e12, 100*t/p))
print()
print("=== 同题三路线: N=65536, d=128, 单头 ===")
N, d = 65536, 128
print("exact(FA):  FLOPs=%.2e  中间显存 naive=%.1fGB -> FA=%.1fMB  精确" % (4*N*N*d, N*N*2/1e9, N*d*2*3/1e6))
w = 4096
print("滑窗/稀疏:  FLOPs=%.2e (省 %.0fx)  近似(截断)" % (4*N*w*d, N/w))
print("线性attn:   FLOPs=%.2e (省 %.0fx)  近似(核重定义)" % (2*N*d*d, (4*N*N*d)/(2*N*d*d)))
