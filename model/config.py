"""模型超参数：维度、层数、注意力头和词表。"""
# 拆分自 Datawhale Happy-LLM 第五章；保留原计算逻辑。许可见项目 LICENSE.txt。
from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768,
            n_layers: int = 12,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            pad_token_id: int = 0,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.pad_token_id = pad_token_id
        super().__init__(**kwargs)

