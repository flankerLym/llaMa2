"""完整语言模型：Embedding、Decoder 堆叠、最终归一化、输出头。"""
# 拆分自 Datawhale Happy-LLM 第五章；保留原计算逻辑。许可见项目 LICENSE.txt。
import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .config import ModelConfig
from .norm import RMSNorm
from .rope import precompute_freqs_cis
from .decoder import DecoderLayer
from .generation import TextGenerationMixin

class Transformer(TextGenerationMixin, PreTrainedModel):
    config_class = ModelConfig  # 配置类
    last_loss: Optional[torch.Tensor] # 记录最后一次计算的损失

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight 

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 不分割的模块列表

    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prepare_attention_mask(self, attention_mask: Optional[torch.Tensor], tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            attention_mask = attention_mask[:, 0, 0, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        attention_mask = attention_mask.to(tokens.device)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask > 0
        if attention_mask.shape != tokens.shape:
            raise ValueError(f"attention_mask shape {attention_mask.shape} must match input_ids shape {tokens.shape}")
        return attention_mask

    def _left_pad_by_attention_mask(
            self,
            idx: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            pad_token_id: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if attention_mask is None or attention_mask.all():
            return idx, attention_mask

        bsz = idx.size(0)
        lengths = attention_mask.long().sum(dim=1)
        max_len = max(int(lengths.max().item()), 1)
        packed_idx = idx.new_full((bsz, max_len), pad_token_id)
        packed_mask = attention_mask.new_zeros((bsz, max_len), dtype=torch.bool)

        for row in range(bsz):
            valid_len = int(lengths[row].item())
            if valid_len <= 0:
                continue
            valid_tokens = idx[row][attention_mask[row]]
            packed_idx[row, max_len - valid_len:] = valid_tokens
            packed_mask[row, max_len - valid_len:] = True

        return packed_idx, packed_mask
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], 输入 token 张量。
        - targets: Optional[torch.Tensor], 目标 token 张量。
        - kv_cache: bool, 是否使用键值缓存。
        - kwargs: 其他关键字参数。

        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        """

        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'labels' in kwargs:
            targets = kwargs['labels']
        attention_mask = self._prepare_attention_mask(kwargs.get('attention_mask'), tokens)

        # 前向传播函数
        _bsz, seqlen = tokens.shape
        # 通过词嵌入层和Dropout层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过Decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, attention_mask=attention_mask)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            ignore_index = self.args.pad_token_id if self.args.pad_token_id is not None else 0
            if torch.any(targets == -100):
                ignore_index = -100
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
                reduction='none'
            )
        else:
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            if attention_mask is None:
                logits = self.output(h[:, [-1], :])
            else:
                full_logits = self.output(h)
                last_token_pos = attention_mask.long().sum(dim=1).clamp(min=1) - 1
                logits = full_logits[torch.arange(_bsz, device=tokens.device), last_token_pos].unsqueeze(1)
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    
