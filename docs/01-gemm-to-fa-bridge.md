# GEMM 十级阶梯 ↔ FlashAttention：同一套方法论的对应表

> 本仓库的核心文档。siboehm 十级阶梯是 kernel 优化方法论的 **GEMM 教学版**，FlashAttention 是同一方法论的 **attention 工业版**——本表把两边逐级对上号，并指出阶梯里"缺"的那一级。

## 一、逐级对应表

| siboehm 级 | 手撕复现数字 | FlashAttention 里的对应物 |
|---|---|---|
| v1 naive | ~1% cuBLAS（AI=0.25） | 标准 attention 三步物化（同病：没管搬运，N×N 三进三出 HBM） |
| v2 线程重排 / coalescing | ~8%（带宽×7） | FA 读 KV 块的合并访问（warp 逐块读） |
| v3 SMEM tiling | ~13%（global 读÷32） | **FA1 的 tiling**：K/V 块进 SRAM，中间矩阵永不落地 |
| v4/v5 寄存器分块（累加器驻寄存器） | ~36% → ~69% | FA 的 **(m, ℓ, O) 三元组全程驻寄存器**——E 系列的流式状态就是 v4 `threadResults[TM]` 的 attention 版 |
| v6 float4 向量化 | +9.7 → 78.4% | FA3 的 TMA 整块搬运（更粗粒度的"一次搬一大口"） |
| v7/v8 bank conflict / padding | +1~2 点（比值法验证） | FA 的 SMEM tile 布局（布局学问通用） |
| v9 autotuning | 84.8% | FA kernel 的 tile 尺寸调参（同款思想） |
| v10 warp tiling / 分工 | **93.7% = FP32 之巅** | **FA2 的 warp 按行分工**（同一级！） |
| v11 double buffering | 回退 74.3%（SMEM×2 → 驻块 6→2） | FA3 的多缓冲 pingpong + 异步预取——**用 TMA 把 v11 的教训救回来**（专用搬数引擎不占 SMEM 预算） |

## 二、阶梯里"缺"的那一级：换算律

对照之后自然要问：**为什么 siboehm 阶梯里没有"换算律"这一级？**

```
GEMM 的累加 acc += A·B 是纯加法——加法天然可结合，
  块怎么切都能累，tiling 直接可用，不需要新钥匙。

attention 的 softmax 归一化跨整行——分子在块内、
  分母在全局，普通 tiling 卡死在这里。

换算律 = 把"不可结合的行归一化"改写成
  "可结合的 (m, ℓ, O) 折叠"：

    m   = max(m_old, m_new)
    ℓ   = α·ℓ_old + ℓ_new           α = exp(m_old − m)
    O   = (α·ℓ_old·O_old + ℓ_new·O_new) / ℓ

  折价链可坍缩（e^{m₁−m₂}·e^{m₂−m₃} = e^{m₁−m₃}），
  所以合并算子满足结合律——任意切法、任意顺序都精确。
```

**结论**：GEMM 不需要换算律不是它更简单，而是它没这个病——attention = tiling 方法论 + 一把只为 softmax 而生的代数钥匙。

验证：`python experiments/01_online_softmax_blocks.py`（分块与直算差 4.4e-16）。

## 三、一句话（面试可背）

> "siboehm 十级阶梯是这套 kernel 优化方法论的 GEMM 教学版，FlashAttention 是它的 attention 工业版——合并访问、SMEM tiling、寄存器驻留、warp 分工、双缓冲，每一级都能对上号；FA 唯一多出来的是换算律这把代数钥匙，因为 GEMM 的加法累加天然可结合，而 softmax 的行归一化不结合——钥匙只为 attention 的病而生。"
