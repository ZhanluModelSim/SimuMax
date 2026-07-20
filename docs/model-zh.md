<p align="center">
  <a href="model.md">English</a>|
  <a href="model-zh.md">中文版本</a>
</p>

# Model 配置

SimuMax 的模型配置文件位于 [configs/models](../configs/models)。model 文件描述的是模型的静态结构，是 cost、memory 和 simulator 建模的基础。

SimuMax 依赖三个输入文件共同工作：

- **system**：机器能力与效率数据
- **strategy**：并行与运行策略
- **model**：模型结构描述

相关文档：

- [system.md](./system.md)
- [strategy.md](./strategy.md)

## 最快起步方式

除非非常特殊，否则不要从空文件开始写。

推荐路径：

1. 从 [configs/models](../configs/models) 复制最接近的已有 JSON。
2. 保留原文件作为对照。
3. 只改结构上真正不同的字段。
4. 先配一个已知可用的 `strategy` 和 `system` 跑 `perf`。

常见起点：

- dense 模型：
  [configs/models/llama3-8b.json](../configs/models/llama3-8b.json)
- MoE + MLA 模型：
  [configs/models/deepseekv2.json](../configs/models/deepseekv2.json)

## 最小可用 dense model 示例

```json
{
    "model_type": "dense",
    "model_name": "my_dense_model",
    "hidden_size": 4096,
    "head_num": 32,
    "kv_head_num": 8,
    "head_size": 128,
    "intermediate_size": 14336,
    "layer_num": 32,
    "vocab_size": 128256,
    "use_swiglu": true
}
```

第一次改 dense 模型时，最短路径通常是：

1. 复制 `llama3-8b.json`
2. 修改 `model_name`
3. 修改 `layer_num`、`hidden_size`、`head_num`、`kv_head_num`、`intermediate_size`、`vocab_size`
4. 只有在不是标准 MHA 时，再改 `attention_type`

## 哪些字段必须和真实模型对齐

至少要和目标 Megatron / 真实模型对齐：

- `layer_num`
- `hidden_size`
- `head_num`
- `kv_head_num`
- `intermediate_size`
- `vocab_size`
- `attention_type`
- `use_swiglu`

如果这些字段不对，即使 `strategy` 和 `system` 对齐，timing 和 memory 也会明显漂移。

## 核心字段

常见字段包括：

- `model_name`
- `layer_num`
- `hidden_size`
- `head_num`
- `kv_head_num`
- `intermediate_size`
- `vocab_size`
- `use_swiglu`
- `attention_type`

这些字段决定了 dense attention / MLP 的主要 shape，以及 embedding / LM head 的规模。

## MoE 与 MLA 字段

MoE 相关字段：

- `expert_num`
- `topk`
- `moe_ffn_hidden_size`
- `moe_shared_expert_intermediate_size`
- `dense_layers`
- `capacity`
- `moe_pad_expert_input_to_capacity`
- `group_linear_mode`

其中 `dense_layers` 很重要：

- 它表示当前 MoE 模型前面有多少层仍然按 dense layer 建模。
- 这个字段会直接影响 pipeline stage 的 memory 和 timing。

MLA 相关字段：

- `attention_type="mla"`
- `qk_head_dim`
- `qk_pos_emb_head_dim`
- `v_head_dim`
- `q_lora_rank`
- `kv_lora_rank`

MoE / MLA 用户建议重点检查：

- MoE:
  - `expert_num`
  - `topk`
  - `moe_ffn_hidden_size`
  - `moe_shared_expert_intermediate_size`
  - `dense_layers`
- MLA:
  - `attention_type="mla"`
  - `qk_head_dim`
  - `qk_pos_emb_head_dim`
  - `v_head_dim`
  - `q_lora_rank`
  - `kv_lora_rank`

## recipe（可选）

可选的层组合声明（
[design_simu_cost_model_tunability-zh.md](./design_simu_cost_model_tunability-zh.md)
Phase 3）。model JSON 可以不再只依赖 `layer_num` / `dense_layers`，
而是用已注册模板声明 block 堆叠：

```json
"recipe": {
    "blocks": [
        {"template": "DenseLLMBlock", "count": 3},
        {"template": "MoELLMBlock", "count": 58}
    ]
}
```

v1 模板：

| 模板 | block 家族 |
|---|---|
| `DenseLLMBlock` | dense transformer 层 |
| `MoELLMBlock` | MoE transformer 层 |

展开与校验规则：

- `layer_num` 取所有 block `count` 之和；`dense_layers` 取前缀
  `DenseLLMBlock` 的数量。
- Dense block 必须构成前缀：`MoELLMBlock` 之后再出现
  `DenseLLMBlock` 会触发校验错误。
- 当 `recipe` 与显式的 `layer_num` / `dense_layers` 冲突时，以
  recipe 为准并打印 warning。
- 不配置 `recipe` 时行为不变——现有 `layer_num` + `dense_layers`
  模式即隐式默认配方。

模板注册在 cost spec 注册表（`simumax/core/cost_specs.py`）中，注册表
把每个模板映射到其 block 家族与算子绑定；成本公式本身搬入注册表是
文档记录的后续工作。可运行示例见
[configs/models/llama2-tiny-recipe.json](../configs/models/llama2-tiny-recipe.json)——
用 recipe 表达的 `llama2-tiny`。

## 词表 padding

与 Megatron 对齐时，词表 padding 相关字段包括：

- `make_vocab_size_divisible_by`
- `padded_vocab_size`
- `orig_vocab_size`

这些字段会影响 CE、embedding 和 LM head 的 shape 对齐。

## 与 Megatron 的对应关系

如果要和 Megatron real run 对齐，model 配置至少要和真实模型保持一致：

- 层数
- dense / MoE 布局
- MLA / MHA 类型
- expert 数量与 top-k
- MLA 的 LoRA rank
- 词表大小与 padding 行为

如果这些字段不一致，即使 strategy 和 system 对齐，timing 和 memory 也可能明显漂移。

## 典型使用流程

1. 从 [configs/models](../configs/models) 里找最接近的模型。
2. 复制并修改结构字段。
3. 和对应的 strategy / system 一起使用。
4. 先跑 `perf`，需要 trace 或 memory 生命周期证据时再跑 `simulate()`。

## 示例

```python
from simumax.core.config import ModelConfig

model = ModelConfig.init_from_config_file("configs/models/llama3-8b.json")
print(model.layer_num, model.hidden_size, model.model_type)
```
