# Happy-LLM 第五章：从创建项目到 Kaggle GPU

这是第五章的 **Kaggle 小规模教学适配版**，不是原作者 215M 模型效果复现，也不是 Qwen 微调。模型从随机权重开始；不需要下载模型权重。

完整流程：准备中文文本 → 自训 BPE 分词器 → 手写 Decoder 模型 → next-token 预训练 → 全参数 SFT → 留出集评估 → 文本生成。

## 1. 先创建本地项目（Windows PowerShell）

将压缩包解压到 `D:\Project`，得到 `D:\Project\happy_llm_kaggle`。该文件夹就是完整项目，不需要再套一层文件夹。

```powershell
cd D:\Project\happy_llm_kaggle
uv python install 3.11
uv venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

本地只安装 CPU PyTorch，不消耗本地 GPU。下面明确使用虚拟环境的 Python，不依赖激活脚本。依赖下载失败时先检查网络，不要安装到全局环境。

## 2. 本地先跑通微型流程

`examples` 是自带的重复玩具文本，仅用于检查代码，训练/验证/测试内容存在重合，不能用于报告模型能力。

```powershell
.venv\Scripts\python.exe train_tokenizer.py --data examples/pretrain_train.jsonl --out smoke_tokenizer --vocab-size 512
.venv\Scripts\python.exe check.py --tokenizer smoke_tokenizer
.venv\Scripts\python.exe train.py --stage pretrain --config configs/smoke.json --data-dir examples --tokenizer smoke_tokenizer --out smoke_outputs/pretrain --steps 3 --batch-size 2 --accum 2 --eval-every 1 --cpu
.venv\Scripts\python.exe train.py --stage sft --config configs/smoke.json --data-dir examples --tokenizer smoke_tokenizer --init smoke_outputs/pretrain/last.pt --out smoke_outputs/sft --steps 3 --batch-size 2 --accum 2 --eval-every 1 --cpu
.venv\Scripts\python.exe generate.py --checkpoint smoke_outputs/sft/best.pt --tokenizer smoke_tokenizer --chat --prompt 你好 --max-new-tokens 16
```

成功标准：打印 PASS、loss 有限、两个阶段各有 `last.pt` 和 `best.pt`、生成脚本可以输出 JSON。短训练后乱码或重复很正常，成功标准不是回答质量。

## 3. 在 Kaggle 上运行

1. 登录 https://www.kaggle.com/code ，创建 Notebook。
2. 使用 Notebook 的导入功能导入本包的 `kaggle_run.ipynb`（菜单名称可能随界面调整）。
3. 在 Notebook 设置里选择当前可用的 GPU 加速器，开启 Internet。GPU 是否可选、验证要求及额度以你的账户界面为准。
4. **按单元格顺序逐个执行**，不要一开始就整本运行。第一格内嵌全部项目源码，自动解压到 `/kaggle/working/happy_llm_kaggle`，不用额外上传 zip 或创建代码 Dataset。
5. 第二格安装 Python 依赖，保留 Kaggle 自带的 CUDA PyTorch。第三格检查 GPU；如果检查失败，先解决加速器设置再继续。
6. 第四格抽取真实公开数据：预训练 20,000 条、SFT 5,000 条。接着训练 tokenizer、检查模型、预训练、SFT、评估和生成。
7. 最后一格把源码、数据、分词器、权重、日志、图片打成 `/kaggle/working/happy_llm_result.zip`。下载该文件；使用 Kaggle 的保存版本/运行功能保存本次输出，避免临时会话结束丢失文件。

默认 **只使用一张 GPU**，即使分配了两张也只用 `cuda:0`；这里先学习完整流程，不引入多卡 DDP。依赖安装后如已有导入旧版 transformers，重启会话并重跑前几格。

## 4. 每个脚本负责什么

| 文件 | 学习内容 | 产物 |
|---|---|---|
| `prepare.py` | 流式抽样、去重、固定种子划分 90%/5%/5% | `data/*.jsonl`、清单和哈希 |
| `train_tokenizer.py` | ByteLevel BPE、词表、特殊 token | `tokenizer/` |
| `model/` | RMSNorm、RoPE、GQA、SwiGLU、残差、权重共享 | 模型结构 |
| `data.py` | 输入与标签错位、padding、SFT 回答掩码 | X、Y |
| `train.py` | 交叉熵、反传、AdamW、FP16、梯度累积、验证 | checkpoint、CSV |
| `evaluate.py` | 全部留出测试集 token loss/PPL | 测试 JSON |
| `generate.py` | 逐 token 自回归生成 | 文本、单次生成耗时 |
| `check.py` | 因果注意力、回答标签、模板和权重共享检查 | PASS |
| `plot.py` | 训练与验证曲线 | `outputs/loss.png` |

阅读顺序建议：先 `data.py` 搞清楚 X/Y，再按 `model/transformer.py` → `decoder.py` → `attention.py` → `rope.py` → `ffn.py` → `norm.py` 阅读，最后读训练循环。

## 5. 数据和模型设置

原章使用序列猴子与 BELLE。本版默认用 MiniMind 提供的两份 JSONL，固定到 `312afb4` 修订版本，流式读取前 N 条有效、不重复的数据后关闭连接，不下载完整文件。原始文件约 1.24GB 与 1.74GB，实际抽样大小取决于文本长度；流式请求可能有额外网络预读。

预训练格式：`{"text":"一段中文文本"}`。SFT 格式：`{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"答案"}]}`。也支持 MiniMind 的 `conversations` 和 BELLE 的 `instruction/input/output`。工具调用样本会过滤；本项目只训练普通文本对话，不训练工具或独立思考字段。

仅在各阶段内部按完整样本去重；抽样是前缀抽样，不代表全数据的随机样本，也不保证预训练语料与 SFT 测试内容没有语义重合。数字用于本项目流程比较，不是公开标准 benchmark。

小模型配置：8 层、512 隐藏维度、8 个 Q 头、4 个 KV 头、上下文 256 token、词表目标 6144，参数量约 26.75M（实际以日志为准）。这是稠密 FFN 模型，**没有 MoE**。

默认预训练 500 个优化器更新步，SFT 200 步。每步 4 条 × 梯度累积 4 次，即有效 batch 16。验证默认只用前 20 个 batch，最终测试用全部有效测试样本。SFT 只计算回答及结束 token 的损失；截断后无回答的样本会过滤，数量会打印。

从零小模型经过这么少的数据和训练，通常仍会胡言乱语。不要据此判断结构有问题，也不要把玩具数据的 loss 当成实验结论。想提升回答质量，需要更多合适数据与训练步数。

## 6. 扩大训练、显存不足与续训

跑通后可以把抽样改到 100,000/20,000 条、预训练改到 3,000 步、SFT 改到 1,000 步；是否够时间需用实测步速估计，不承诺免费额度能完成。重新训练 tokenizer 后，旧模型权重不能沿用。

显存不足：先把 `--batch-size 4 --accum 4` 改成 `--batch-size 2 --accum 8` 或 `1/16`。降低序列长度会改变模型配置，必须在开训前修改，已有 checkpoint 不应混用。

同阶段断点恢复（保持数据、tokenizer、config、steps、batch、accum、lr 相同）：

```bash
python train.py --stage pretrain --resume outputs/pretrain/last.pt --steps 500 --batch-size 4 --accum 4
```

SFT 恢复用 `--stage sft --resume outputs/sft/last.pt --steps 200 --lr 1e-4`，不再传 `--init`。默认每 100 步保存；中断最多丢失尚未保存的这段进度。恢复会还原优化器、scaler、随机状态与数据 batch 位置；CUDA 算子的非确定性仍可能造成细小数值差异。

跨会话要先把保存的结果包重新加入 Kaggle 输入并解压到工作目录。保持原来的分词器和数据。想延长训练：用 `--init outputs/pretrain/last.pt --out outputs/pretrain_more --steps 1000` 开新轮；这是继承权重并重新初始化优化器和调度器，不是原进度续训。

离线数据：下载你有权使用的 JSONL 后，通过 `--pretrain-source 本地路径 --sft-source 本地路径` 替换。原章序列猴子要先解压成 JSONL；BELLE 可直接转换。数据下载失败不会静默退回玩具数据。

## 7. 如何看结果

- Loss：预测正确 token 的交叉熵，越低越好。
- PPL：`exp(loss)`，相同 tokenizer、数据、截断规则下越低越好。预训练与 SFT 的掩码/数据不同，不直接互比 PPL。
- Notebook 会在同一个 SFT 测试集上评估 base 和 SFT，便于比较微调前后回答 token 的预测效果。不要承诺一定提升。
- `tokens_per_sec` 的训练日志统计参与损失的 token；SFT 与预训练不能拿它直接比较吞吐。
- 生成脚本没有 KV Cache，单次耗时仅用于观察，不是部署性能结论。
- 推理输出可能有事实错误。学习项目不代表成熟模型能力。

## 8. 来源与改动

模型模块拆分自 Datawhale Happy-LLM `docs/chapter5/code/k_model.py`，提交 `1e0b3a1e6f8936b8ba4bbd0673e8e2227d196c96`，原作者中文注释保留；按模块拆分、生成方法移至 mixin、移除原文件末尾硬编码示例；未改变模型算法或 state_dict 参数名。已有权重可继续加载。新写/改编的外围脚本将原章流程适配为小模型、单 GPU、本地 CPU smoke、独立 pad token、tokenizer往返检查、回答掩码、验证/测试、断点和Notebook。

- 教程：https://datawhalechina.github.io/happy-llm/#/./chapter5/
- 源码：https://github.com/datawhalechina/happy-llm
- 默认数据：https://huggingface.co/datasets/jingyaogong/minimind_dataset
- Kaggle 文档：https://www.kaggle.com/docs/notebooks

随包保留 Happy-LLM 的 CC BY-NC-SA 4.0 许可证；该教学适配包采用同协议。默认数据遵守其来源许可（数据卡列 Apache-2.0 / CC-BY-NC-2.0），未在包中再分发真实语料。

## 9. 验证边界

交付前执行了 CPU 微型预训练、SFT、测试集评估、文本生成及结构检查；真实数据远程入口也已检查。没有使用你的 Kaggle 账户上传、没有实际跑 Kaggle GPU，不提供虚构的 GPU 耗时、显存或模型效果。你导入 Notebook 后，GPU 检查格与训练日志会给出你的实测结果。

## 10. 模型目录与数据流

| 文件 | 对应模型结构 |
|---|---|
| `model/config.py` | 超参数配置 |
| `model/norm.py` | RMSNorm |
| `model/rope.py` | RoPE 位置编码 |
| `model/attention.py` | GQA 因果自注意力 |
| `model/ffn.py` | SwiGLU FFN |
| `model/decoder.py` | 注意力、FFN、归一化和两条残差 |
| `model/transformer.py` | Embedding、8层堆叠、最终归一化、输出头 |
| `model/generation.py` | 自回归采样和生成策略 |
| `model/__init__.py` | 训练/评估脚本统一导入入口 |

模型的数据流：token 编号 → `transformer.py` 的 Embedding → 8个 `decoder.py` → 最终 RMSNorm → 输出头 logits。

单层内部：RMSNorm → attention（Q/K/V，调用 RoPE）→ 残差相加 → RMSNorm → FFN → 残差相加。

`generation.py` 调用完整模型来反复预测下一个 token；它不是额外的网络层。Embedding 和输出头是很短的标准 Linear/Embedding 定义，保留在完整模型装配文件中。

请解压到新目录后使用，避免旧目录中残留 `model.py`。全部训练命令保持原样，Kaggle Notebook 已更新为内嵌拆分后的目录。
