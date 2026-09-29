# -*- coding: utf-8 -*-
"""三地基数值验证: (1)/sqrt(d) 方差归一 (2)RoPE 内积只依赖位置差 (3)MLA 两段式吸收.
对应 docs/02 与 docs/03."""
import numpy as np
np.random.seed(0)

print("=== (1) 为什么除 sqrt(d): Var(q·k) = d ===")
for d in (64, 1024, 4096):
    Q = np.random.randn(2000, d); K = np.random.randn(2000, d)
    dots = np.sum(Q * K, axis=1)
    print("d=%5d: Var=%7.1f (理论 d)  除后 %5.2f (理论 1)" % (d, dots.var(), (dots/np.sqrt(d)).var()))
x = np.random.randn(8) * np.sqrt(4096)
p = np.exp(x) / np.exp(x).sum()
print("不除时(d=4096) softmax 最大概率=%.3f -> 近 one-hot, 梯度消失\n" % p.max())

print("=== (2) RoPE: 内积只依赖位置差 ===")
def rot(v, ang):
    c, s_ = np.cos(ang), np.sin(ang)
    return np.array([c*v[0]-s_*v[1], s_*v[0]+c*v[1]])
th = 0.4; q = np.array([1.0, 2.0]); k = np.array([0.5, -1.5])
for mm, nn in [(2, 5), (7, 10), (0, 3), (100, 103)]:
    print("pos=(%3d,%3d) 差=%3d  内积=%.6f" % (mm, nn, nn-mm, rot(q, mm*th) @ rot(k, nn*th)))
print("-> 差相同则内积相同: 绝对形式编码相对语义\n")

print("=== (3) MLA 两段式分数: absorb 恒等 ===")
d_nope, d_rope, d_lat = 128, 64, 512
W_UK = np.random.randn(d_nope, d_lat)
q_nope, q_rope = np.random.randn(d_nope), np.random.randn(d_rope)
c, k_rope = np.random.randn(d_lat), np.random.randn(d_rope)
s_unabs = q_nope @ (W_UK @ c) + q_rope @ k_rope
s_abs = (q_nope @ W_UK) @ c + q_rope @ k_rope
print("unabsorbed=%.6f absorbed=%.6f diff=%.2e (rope 段无可吸, 原样保留)" % (s_unabs, s_abs, abs(s_unabs-s_abs)))
