# FlashAttention 全景：三件套、三代演进、三卡对照

## 一、FA 改进了什么（vs 2017 原始 attention）

公式 `O = softmax(QKᵀ/√d)·V` 到今天一字未变——**FA 改进的是执行路径，不是数学**：

| 维度 | 2017 原始版 | FlashAttention |
|---|---|---|
| 数学公式 | O=softmax(QKᵀ/√d)V | 一字不变（exact） |
| S/P 是否物化 | 是（N=64K 单头 8.6GB） | 永不物化（tile 内活） |
| kernel 数 | 3 算子三进三出 | 1 个融合 kernel |
| 中间显存 | Θ(N²) | Θ(N) |
| HBM 访问 | Θ(Nd+N²) | Θ(N²d²/M)，**下界匹配** |

三件套：**①tiling**（治落地）→ **②online softmax 换算律**（治归一化跨行）→ **③融合**（治三进三出）。反向不存 S/P，只存 O 和 L（logsumexp），重算重建。

## 二、三代演进：每代解决上一代的什么

| | FA1 (2022) | FA2 (2023) | FA3 (2024) |
|---|---|---|---|
| 解决 | 不落地 | 吃满 GPU | 与硬件共生 |
| 核心创新 | tiling+换算律+融合 | 序列维并行/warp 按行分工 | warp 专业化/TMA/FP8 |
| IO 阶 | Θ(N²d²/M) | 同阶（降常数） | 同阶（降常数） |
| 利用率 | 25-40% (A100) | **72%**（225 TFLOPs） | **75% FP16**（740T）/ FP8 ~1.2 PFLOPs |
| 硬件门槛 | Ampere 即可 | Ampere 即可 | **仅 Hopper (sm90)** |

规律：**数学代代不变、IO 阶已封底（Prop 3 下界）——三代全在常数上做文章，武器从算法（FA1）到调度（FA2）到硬件特性（FA3）。**

FA3 三招对比原来：①人人同活分时串行 → 生产者/消费者 warp 专业化；②每线程 cp.async → TMA 整块下单（独立 DMA 引擎）；③fp16 一条路 → FP8 + 非相干处理（block 量化+随机旋转；attention 天生适合 FP8，因为 softmax 有 m 折价锚管住动态范围）。

## 三、三卡对照：同一思想的两种落地（NVIDIA/昇腾/寒武纪）

| 维度 | NVIDIA Hopper | 昇腾（达芬奇） | 寒武纪 MLU |
|---|---|---|---|
| 矩阵单元 | Tensor Core (WGMMA) | AI Cube (AIC) | MLUcore |
| 向量单元 | 与矩阵同核（**FA3 用软件分离**） | AI Vector (AIV) 独立 | Vector 单元 |
| 搬数引擎 | TMA（Hopper 新增） | MTE1/2/3 三引擎 | DMA |
| "分工"层次 | 软件模拟分工 | **硬件原生分工** | 硬件原生分工 |

读法：**FA3 用 warp 专业化模拟的"计算/访存/向量分工"，达芬奇架构在 ISA 层原生就有**——不是国产卡学 FA3，而是 FA3 把 GPU 向达芬奇式架构推进了一步。国产卡的课题是用软件栈（CANN/MindIE、CNNL）把原生分工兑现成 attention kernel。

## 四、与 PagedAttention 的关系（正交）

- **FlashAttention**：attention 计算的 kernel 级优化（算得快）
- **PagedAttention**：KV cache 的内存管理（存得省）
- 会师点：FA 族变体接受 block table 参数——kernel 顺页表取散放 KV 块，逐块进 tiling 流水。两个"块"是不同的块（paged 块=显存管理单元 16 token；tile=计算分块）。
