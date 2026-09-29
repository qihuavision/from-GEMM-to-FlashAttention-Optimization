# 面试口述稿：attention 优化的演进（60 秒版 + 3 分钟版 + 追问树）

## 60 秒版（背到肌肉记忆）

> "attention 的公式 2017 年至今没变过，变的全部在执行层。原始实现把 softmax(QKᵀ/√d)V 分三步算，N×N 的中间矩阵三进三出显存——64K 上下文单头就要 8 个 GB，这是 IO 瓶颈不是算力瓶颈。
>
> 治它的思想 2018 年就有了：NVIDIA 的 online softmax 把归一化改成流式维护 max 和分母；2021 年 Rabe & Staats 把它用于 attention 做到根号 N 显存。2022 年 FlashAttention 完成系统化：tiling 切块进 SRAM、换算律块间合并、一趟融合 kernel——中间矩阵永不落地，显存从 N 平方降到 N，还给了 IO 复杂度定理：N²d² 除以 SRAM 大小，并且证明了这个阶的下界——它是渐近最优的。
>
> 之后两代都在降常数：FA2 重排并行度和 warp 分工，A100 上跑到 225 TFLOPs、72% 利用率；FA3 用满 Hopper 的 TMA、warp 专用化和 FP8，H100 上 740 TFLOPs、75%，FP8 接近 1.2P——但它只在 sm90 上有效，A100 回退 FA2。
>
> 另一条线是 KV 压缩：MQA 共享单写头、GQA 分组共享还能用 5% 算力把存量模型转化过来、DeepSeek 的 MLA 干脆把 KV 压进 512 维 latent 加 64 维解耦 RoPE——每 token 只 70KB，配自家 FlashMLA 内核把 decode 带宽打满。数学层 FA 系是精确的、零质量损失，这就是它赢过稀疏和线性路线的原因。"

## 3 分钟版追加两段（60 秒版讲完没被打断时）

> "再往深一层我推过三个东西：一是分块合并的结合律——折价因子链可以坍缩，所以任意切法任意顺序都精确，这是 tiling 的数学许可证；二是反向传播只需要存输出和 logsumexp，概率矩阵用它们现场重建；三是 MLA 为什么必须解耦 RoPE——旋转是位置相关的，和投影矩阵不可交换，压进共享 latent 数学上不成立，所以 576=512+64 是唯一解。
>
> 落到国产这条线上，工程分四层：模型层 MLA、NSA 稀疏省算力；kernel 层 FlashMLA 把 decode 打到约 3000GB/s；引擎层 SGLang 的 radix 树做前缀 KV 复用、FlashInfer 统一变长 kernel；系统层 Mooncake 把 KV 池搬出 GPU 做存算分离。每一层都是对一个具体瓶颈的定向爆破——瓶颈清单就是创新清单。"

## 追问树（12 个钩子 → 答案在哪）

| 口述里的钩子 | 追问 | 答案位置 |
|---|---|---|
| "IO 瓶颈不是算力" | 为什么？ | docs/02 对照表；IO 定理 |
| "8 个 GB" | 怎么算？ | N²×2B（N=64K 单头） |
| "online softmax 2018" | 是 FA 发明的吗？ | 不是——Milakov & Gimelshein 2018（arXiv 1805.02867）；MEA=Rabe & Staats 2021（arXiv 2112.05682）；FA 的原创=融合实现+定理 |
| "换算律流式维护" | 为什么精确？ | 结合律：折价链坍缩 e^{m₁−m₂}·e^{m₂−m₃}=e^{m₁−m₃}；experiments/01 实测 4.4e-16 |
| "IO 定理+下界" | 推一遍？ | Thm2/Prop3（arXiv 2205.14135）：标准 Θ(Nd+N²) vs FA Θ(N²d²/M)；手推 B_r≈M/d ⇒ 总 IO=(N/B_r)·2Nd |
| "72%/75% 利用率" | 哪来的？ | experiments/04：225/312、740/989 |
| "只在 sm90" | 为什么？ | WGMMA/TMA/FP8 三件武器全是 Hopper 硬件；A100 回退 FA2 |
| "A100 回退 FA2" | 国产卡呢？ | docs/02 三卡对照；移植三选一（Triton/CUTLASS 同位库/原生 ISA） |
| "GQA 5% 算力转化" | 怎么转？ | uptraining：组内均值池化 K/V + ~5% 预训练算力微调（arXiv 2305.13245） |
| "MLA 70KB/token" | 怎么算？ | experiments/03：(512+64)×61×2B；三本账 3.9MB/244KB/70KB |
| "为什么解耦 RoPE" | 证明？ | R·W ≠ W·R（2×2 手例差 0.1646）——docs/03 |
| "decode 3000GB/s" | 为什么是带宽界？ | absorbed 形态=576 维 MQA，读 576 vs 32768=57×；prefill 算力界用 unabsorbed |

## 反陷阱五连（判词）

1. "online softmax 是 FA 发明的吧？" → **不是**，2018 NVIDIA；FA 贡献=融合+定理
2. "FA 是近似算法吧？" → **exact**，结合律保证（实验 4.4e-16）
3. "MLA 是 GQA 的一种吧？" → **不同轴**：GQA 删份数（离散），MLA 低秩换坐标系（连续）+解耦 RoPE
4. "除 √d 是经验超参吧？" → **精确方差归一**：Var(q·k)=d，experiments/02 实测 64.4/1008.8/4232
5. "FA3 出来 FA2 淘汰了吧？" → FA3 绑定 sm90；A100/国产卡/AMD 生产主力仍是 FA2/Triton——按硬件选 kernel
