# -*- coding: utf-8 -*-
"""MHA/GQA/MLA 每 token KV cache 三本账 (DeepSeek-V3 口径: 61 层, bf16, head_dim=128).
MLA = kv_lora_rank 512 + rope 64. 对应 docs/03."""
L, hd, bf = 61, 128, 2
mha = 2 * L * 128 * hd * bf          # 128 个 KV 头
gqa = 2 * L * 8 * hd * bf            # 8 个 KV 头 (Llama-3 档)
mla = L * (512 + 64) * bf            # 压缩 latent + 解耦 rope
for n, b in [("MHA(128头)", mha), ("GQA(8头)", gqa), ("MLA(512+64)", mla)]:
    print("%-14s %.0f KB/token   40GB 池可装 %.0f 万 token" % (n, b / 1024, 40e9 / b / 1e4))
print("MLA = GQA 的 %.0f%%, MHA 的 %.1f%%  (官方口径 70KB/token 对上)" % (100*mla/gqa, 100*mla/mha))
