# 交付验证记录

执行环境：Python 3.12.14、PyTorch 2.14.1+cpu、Transformers 4.46.3、Tokenizers 0.20.3。无 CUDA GPU。

已执行：

- 所有 Python 脚本语法编译。
- BPE 训练和中文/英文/特殊 token 的精确编解码往返检查。
- 因果性：改变后半段 token，前半段 logits 保持不变。
- SFT 掩码：只有回答及结束 token 作为目标，用户文本/角色头/padding 被忽略。
- 训练拼接与 tokenizer chat_template 的 token 序列一致。
- Embedding 与输出层权重共享检查。
- 自带玩具数据：微型模型 CPU 预训练 3 步、SFT 3 步、完整留出集评估、生成 8 个 token。
- 玩具数据预训练验证 loss 6.1144 → 5.8563，SFT 验证 loss 6.0425 → 5.8710。玩具数据有重复，不作为模型能力指标。
- 中断在第 1 步后恢复，跑到第 4 步，与连续跑 4 步的最终参数逐项比较，最大绝对差 0.0。
- 实际从固定远程数据版本抽取预训练/SFT 各 20 条，生成划分与哈希清单。
- 真实小样本以 8 层、dim=512、词表=1024 运行 CPU 预训练 1 步和 SFT 1 步，均得到有限 loss 和 checkpoint。一步训练的 loss 不构成能力或泛化结论。
- 默认模型在 vocab=6144 时参数量 26,747,392。

未执行：用户 Kaggle 账户上传、Kaggle GPU 训练、完整默认 500/200 步实验。Notebook 含 GPU 检查和完整运行单元格，需要导入用户账户后执行。

## 模块拆分后验证（本次更新）

- 删除单文件 model.py，按结构拆成 model/ 下 9 个文件。
- 拆分前后：参数名称、初始化参数、前向 logits、贪婪生成和固定种子的随机采样完全一致。
- 拆分前的 state_dict 严格加载成功，没有缺失或额外参数。
- 重新执行 BPE、结构检查、CPU 微型预训练 3 步、SFT 3 步、测试评估、生成。
- Kaggle Notebook 内嵌源码已重新打包，并验证字节一致性和所有 Python 单元格语法。
- 未执行 Kaggle GPU 训练。
