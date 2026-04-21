# PlugDTI Plugin

这是一个可插拔 DTI/DTA 序列增强插件，目标是：

- **使用本地预训练模型** 提取药物 SMILES 和蛋白序列嵌入；
- 输出 **可直接拼接** 到其他 DTI/DTA 主干模型中的向量特征；
- 支持 **冻结编码器、池化、投影降维、磁盘缓存、保存/加载**；
- 提供 **真实本地模型 demo** 和 **可离线跑通的 tiny demo**。


## 1. 目录结构

```text
plugdti_plugin_rebuilt/
├─ plugdti/
│  ├─ __init__.py
│  ├─ cache.py
│  ├─ config.py
│  ├─ fusion.py
│  ├─ plugin.py
│  └─ utils.py
├─ demo/
│  ├─ demo_real_local_models.py
│  └─ demo_tiny_local_models.py
├─ tests/
│  └─ test_smoke.py
├─ requirements.txt
├─ setup.py
└─ README.md
```

---

## 2. 核心设计

### 2.1 插件输入输出

输入：
- drug: SMILES 列表
- protein: 氨基酸序列列表

输出：
- `drug_embeddings`: `[B, Dd]`
- `protein_embeddings`: `[B, Dt]`

其中输出向量可以直接与原主干模型的 `drug feature` 和 `protein feature` 进行拼接。

### 2.2 支持的能力

- 本地加载 `AutoTokenizer` + `AutoModel`
- 药物和蛋白使用不同的本地预训练模型目录
- 池化方式：`mean / cls / max`
- 输出投影：
  - `drug_projection_dim`
  - `protein_projection_dim`
  - 或 `common_projection_dim`
- 冻结编码器（适合插件模式）
- 磁盘缓存 + 内存缓存
- `save_pretrained()` / `from_pretrained()`
- 下游拼接示例头：`ConcatMLPFusionHead`

---

## 3. 安装

先进入项目目录：

```bash
cd plugdti_plugin_rebuilt
```

安装依赖：

```bash
pip install -r requirements.txt
```

或者：

```bash
pip install -e .
```

---

## 4. 最小离线 demo（推荐先跑它）

这个 demo 会：

1. 本地创建两个超小型 BERT 编码器；
2. 用它们模拟“本地预训练药物模型”和“本地预训练蛋白模型”；
3. 提取序列嵌入；
4. 和一个假的下游 backbone feature 做拼接；
5. 跑通保存与重新加载。

运行：

```bash
python demo/demo_tiny_local_models.py
```

预期输出类似：

```text
drug embedding shape: (3, 32)
protein embedding shape: (3, 32)
downstream logits shape: (3, 1)
reloaded protein shape: (2, 32)
```

---

## 5. 真实本地模型 demo

打开文件：

```text
demo/demo_real_local_models.py
```

把这两个路径改成你自己的本地模型目录：

```python
DRUG_MODEL_DIR = r"/path/to/local/chem_model"
PROTEIN_MODEL_DIR = r"/path/to/local/prot_model"
```

然后运行：

```bash
python demo/demo_real_local_models.py
```

---

## 6. 在下游 DTI 主干模型中如何接入

假设你的原始主干已经算出了：

- `backbone_drug_feat`: `[B, Hd]`
- `backbone_protein_feat`: `[B, Hp]`

那么插件输出：

- `plugin_drug_feat`: `[B, Dd]`
- `plugin_protein_feat`: `[B, Dt]`

拼接方式就是：

```python
fusion_input = torch.cat(
    [backbone_drug_feat, backbone_protein_feat, plugin_drug_feat, plugin_protein_feat],
    dim=-1,
)
```

你可以直接接自己原来的 decoder，也可以先看 `plugdti/fusion.py` 里的 `ConcatMLPFusionHead`。

---

## 7. 最常用示例

```python
from plugdti import PlugDTIConfig, PlugDTIPlugin

config = PlugDTIConfig(
    drug_model_dir="/your/local/chembert",
    protein_model_dir="/your/local/protbert",
    drug_max_length=256,
    protein_max_length=1024,
    common_projection_dim=128,
    freeze_encoders=True,
    device="cuda",
)

plugin = PlugDTIPlugin(config).enable_cache("./embed_cache")

out = plugin(
    ["CCO", "CCN(CC)CC"],
    [
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY",
        "MKWVTFISLLLLFSSAYSRGVFRR",
    ],
)

E_d = out["drug_embeddings"]
E_p = out["protein_embeddings"]
print(E_d.shape, E_p.shape)
```

---

## 8. 你后续怎么改最合适

如果你后面要接到basebone里，最稳妥的方式是：

- **插件只负责提取和缓存序列语义特征**；
- **宿主模型负责自己的结构编码与交互模块**；
- 在宿主模型的 `encoder output -> interaction/decoder` 之间做拼接；
- 如维度太大，就把 `common_projection_dim` 设成 64/128/256。

这也是最符合你“即插即用”的论文定位的。

---

## 10. 建议的本地模型

你的场景里通常可以这样放：

- drug model: 本地 ChemBERT / SMILES-BERT / PubChemBERT
- protein model: 本地 ProtBERT / ESM / 其他蛋白序列模型

只要它们能被 `transformers.AutoTokenizer` 和 `transformers.AutoModel` 正常从本地目录加载，这个插件就能直接用。
