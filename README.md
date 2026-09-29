# From GEMM to FlashAttention

**一条主线读懂 LLM attention 优化：从手撕 SGEMM 十级阶梯，到 FlashAttention 三代，再到 MLA 与国产工程。**

> 这个仓库是我的 attention 优化学习线。起点是手撕 siboehm 的 SGEMM 十级优化阶梯（v1 naive → v11 double buffering，亲手复现到 cuBLAS 的 93.7%）；学到 FlashAttention 时发现：**FA 不是发明了新优化技术，它是把同一套 kernel 优化方法论搬进了 attention——唯一真正的新钥匙是一把代数钥匙（online softmax 换算律）**。这个仓库就是这条认知线的沉淀：桥表、演进笔记、可复现的验证实验、面试口述稿。

## 为什么这个仓库值得看

1. **一张表打通两条线**：siboehm GEMM 阶梯 ↔ FlashAttention 逐级对应（见 [docs/01-gemm-to-fa-bridge.md](docs/01-gemm-to-fa-bridge.md)）——v3 SMEM tiling 对应 FA1 的 tiling，v10 warp tiling 对应 FA2 的 warp 分工，v11 double buffering 对应 FA3 的 pingpong……以及阶梯里"缺"的那一级：换算律。
2. **原理不是背的，是算的**：每个关键结论都配一个 10 行以内的验证脚本（`experiments/`），全部实测：分块 softmax 与直算逐位一致（差 4.4e-16）、√d 的方差账（64.4≈d）、RoPE 内积只依赖位置差（1.424203 四组相同）。
3. **面试可背的演进叙事**：60 秒版/3 分钟版口述稿 + 12 个追问钩子的答案指针（见 [notes/60s-pitch.md](notes/60s-pitch.md)）。

## 快速开始

```bash
pip install numpy
python experiments/01_online_softmax_blocks.py     # 分块=直算, 差 4.4e-16
python experiments/02_attention_foundations.py     # √d 方差账 / RoPE 相对性 / MLA 两段式
python experiments/03_mla_kv_ledger.py             # MHA/GQA/MLA KV cache 三本账
python experiments/04_fa_mfu_ledger.py             # FA1/2/3 利用率对账 + 三路线账
```

## 内容地图

| 文档 | 一句话 |
|---|---|
| [docs/01-gemm-to-fa-bridge.md](docs/01-gemm-to-fa-bridge.md) | 核心资产：siboehm 阶梯 ↔ FA 对应表 + 换算律为什么是唯一缺的一级 |
| [docs/02-flashattention-evolution.md](docs/02-flashattention-evolution.md) | FA 全景：三件套 → 三代演进 → 三卡对照（NVIDIA/昇腾/寒武纪） |
| [docs/03-mla-discovery-chain.md](docs/03-mla-discovery-chain.md) | MLA 四幕发现链：MQA(有损)→GQA(半无损)→MLA(结构性冗余) |
| [notes/60s-pitch.md](notes/60s-pitch.md) | 面试口述稿：60 秒版/3 分钟版 + 追问树 |

## 关键数字速查（全部可复现）

| 数字 | 含义 | 出处 |
|---|---|---|
| 4.4e-16 | 分块 softmax vs 直算（FA 是 exact 的机器证明） | exp/01 |
| Θ(N²d²/M) + 下界匹配 | FA 的 IO 复杂度（可证渐近最优） | FA1 论文 Thm2/Prop3 |
| 72% → 75% → FP8 ~1.2 PFLOPs | FA2@A100 → FA3@H100 的利用率阶梯 | exp/04 |
| 70KB / 244KB / 3.9MB | MLA / GQA / MHA 每 token KV cache | exp/03 |
| 57× | MLA 吸收后 decode 的读带宽差 | docs/03 |
| 3000 GB/s | FlashMLA decode（架构创新必须配 kernel 创新） | FlashMLA 官方 |

## 参考文献（本仓库全部论断的一手来源）

- siboehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*（十级阶梯博客 + SGEMM_CUDA 仓库）
- Milakov & Gimelshein 2018, *Online normalizer calculation for softmax* (arXiv:1805.02867)
- Rabe & Staats 2021, *Self-attention Does Not Need O(n²) Memory* (arXiv:2112.05682)
- Dao et al. 2022, *FlashAttention* (arXiv:2205.14135) ｜ Dao 2023, *FlashAttention-2* (arXiv:2307.08691) ｜ Shah et al. 2024, *FlashAttention-3* (arXiv:2407.08608)
- Shazeer 2019, *Fast Transformer Decoding* (arXiv:1911.02150) ｜ Ainslie et al. 2023, *GQA* (arXiv:2305.13245)
- DeepSeek-AI 2024, *DeepSeek-V2*（MLA, arXiv:2405.04434）｜ *DeepSeek-V3*（arXiv:2412.19437）｜ FlashMLA 仓库
- Zheng et al. 2024, *SGLang* (arXiv:2312.07104) ｜ Qin et al. 2024, *Mooncake* (arXiv:2407.00079)

## License

MIT
