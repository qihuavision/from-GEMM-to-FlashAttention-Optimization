# -*- coding: utf-8 -*-
"""FA 的数学许可证: 分块 softmax 合并可结合 => 任意切法/顺序都 exact.
验证: 分块结果与直算逐位一致. 对应 docs/01 (换算律)."""
import numpy as np
np.random.seed(7)

def merge(A, B):
    mA, lA, aA = A; mB, lB, aB = B
    m = max(mA, mB)
    a, b = np.exp(mA - m), np.exp(mB - m)
    return (m, a * lA + b * lB, a * aA + b * aB)

def fold(blocks):
    r = blocks[0]
    for b in blocks[1:]:
        r = merge(r, b)
    return r

def stats(s, v):
    m = s.max(); w = np.exp(s - m)
    return (m, w.sum(), (w * v).sum())

s = np.random.randn(9) * 3; v = np.random.randn(9)
B = [stats(s[i:i+3], v[i:i+3]) for i in (0, 3, 6)]
r1 = fold([fold([B[0], B[1]]), B[2]])
r2 = fold([B[0], fold([B[1], B[2]])])
direct = stats(s, v)
print("(A·B)·C =", np.round(r1, 6))
print("A·(B·C) =", np.round(r2, 6))
print("直算     =", np.round(direct, 6))
assert np.allclose(r1, r2) and np.allclose(r1, direct)
print("=> 结合律成立: 任意切法/顺序 exact\n")

# 贯穿例: s=v=[1,2,3,4] 切两块, 与 E 系列手算对账
s = np.array([1., 2., 3., 4.]); v = np.array([1., 2., 3., 4.])
m1, l1, O1 = stats(s[:2], v[:2]); m2, l2, O2 = stats(s[2:], v[2:])
m, l, o = merge((m1, l1, O1), (m2, l2, O2))
print("块1: m=%.0f l=%.4f O=%.4f | 块2: m=%.0f l=%.4f O=%.4f" % (m1, l1, O1, m2, l2, O2))
print("合并: O = acc/l = %.4f  vs 直算 %.4f  差 %.1e" % (o / l, stats(s, v)[2] / stats(s, v)[1],
      abs(o / l - stats(s, v)[2] / stats(s, v)[1])))
