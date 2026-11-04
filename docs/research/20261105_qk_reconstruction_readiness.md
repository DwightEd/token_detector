# 完整图 residual → Q/K：只读工程可行性

**结论：所需输入在本地，条件性重建在计算上可行；历史Q/K身份及数值资格尚未通过。** 本页不是新方法或执行准入。没有加载tensor/权重、导入Torch/Transformers、执行projection/模型、读取R/GT或计算新指标；仅复用F061既有图资格并核小config/index及官方源码。H020固定分支已停止，不为此补跑其余请求或调3GiB cap。

## 已有输入与明确边界

两份fresh graph00012/00013已有原`input_ids`、全部32层A/V和`residual[0..32]`，BF16原计算以F32精确保存；既有资格证明其BF16格点、有限值、因果mask和残差加法。`residual[l]`是layer l的输入，l=0..31可用于该层input RMSNorm；图slot32是finalnorm前，不能替换成replay hidden32的postnorm语义。无需重新跑整模型取得这些输入，也不用混入incremental采样或旧layer15缓存的浮点状态。

本地8B资产config/index已确认Llama3.1：32层、H4096、32Q/8KV头、head_dim128、eps1e-5、BF16、attention_bias=false。291个index条目覆盖所有32层Q/K/V projection weights，未见Q/K/V bias；四个原shard在本地。config SHA `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`，index SHA `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b`。本轮未读/重哈16GB权重。

重要身份限制：完整图的历史model记录只有size/mtime等信息，没有该次原权重body SHA attestation。当前资产匹配此前另一个capture的权重证明，不能自动转移为这两份完整图的权重身份。相同架构、路径或文件大小都不解决这一点。

## 合理的资格顺序，而非直接宣称“恢复了原Q/K”

1. 另行冻结有限资格合同后，先从safetensors mmap读取原input IDs对应的embedding行，和图residual0比较；无需把约1GB embedding整张装入host。只资格被观测行，不证明全checkpoint相同。
2. 用原`residual[l]`和当前该层input RMSNorm/V权重，调用官方原生BF16路径重建V，并对**全部预定层/坐标**比较既存fresh V。输入是BF16可表示的F32保存值，应恢复原dtype语义；当前H017 FP32 LinearF32 streamer不能未经更改/资格就充当原BF16 producer。先冻结位同一、误差、finite与拒绝规则，不能见差异后拟合容差。
3. embedding/V通过也只约束这些参数/路径；不独立证明Q/K权重。之后才能把同一norm输出代入官方Q/K projections，按同一config的Llama3 RoPE、position IDs、32/8 contiguous GQA、scale和因果mask重建A，逐物理head/原q对已存A做独立closure资格。没有这一步，Q/K只能称当前参数下的条件计算值。
4. 即使A在既定数值预算内相容，也只是operational agreement，不是补造历史权重SHA。A存在行常数/softmax精度等不可识别性，不能从A单独唯一反演raw Q/K。失败须保留并停止语义归因，不能把差异都归为权重或都归为GEMM。

官方4.57.1 `LlamaRMSNorm.forward`先以FP32求方差，再cast回输入dtype并乘norm weight；Q/K/V在RoPE前分别投影，只有Q/K旋转。必须保留原算子顺序、输出cast和mask/softmax路径，不能用一条理想实数公式掩盖中间BF16舍入。两图N730/681不同会改变GEMM形状/算法；只投影几行或统一crop N也可能引入新的舍入合同。精确因果模型的source状态不依赖未来R，但不同full-N执行的保存值不能预设bitwise相等。

## 静态资源与复用范围

N730每层raw residual为11,960,320B F32；native BF16 Q约5,980,160B、K/V各1,495,040B。Q/K/V BF16权重分别32MiB/8MiB/8MiB，单层norm仅8KiB。可使用已审safetensors按tensor mmap/一层驻留思路，host先搬一层raw再释放、A按head流式读取；不用整shard装host、整层32head A全复制或LM head/O。以上只是shape规划，不是实际内存峰值或已通过资源门。任何projection/官方算子执行仍需独立claim及实际资源资格，并单列affine/norm/attention计算成本，不能叫零计算。

已有F040/F041的QK-role诊断、旧O1–O5、source-null/V donor/suffix/ownership/endpoint路径已经执行；本次可行性不能把它们换名为新机制。数值重建资格也不等于新的语义owner、检测risk或文本核验之外的增量。若继续，下一提案须给出区别旧prior的判别问题和必要对照，不能把更多Q/K量本身当创新或重新救H019符号/幅度。
