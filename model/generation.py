"""文本生成策略；不参与模型层堆叠，无 KV Cache。"""
# 拆分自 Datawhale Happy-LLM 第五章；保留原计算逻辑。许可见项目 LICENSE.txt。
from typing import Optional
import torch
import torch.nn.functional as F

class TextGenerationMixin:
    @torch.inference_mode()
    def generate(
            self,
            idx,
            stop_id=None,
            max_new_tokens=256,
            temperature=1.0,
            top_k=None,
            attention_mask: Optional[torch.Tensor] = None,
            pad_token_id: Optional[int] = None
    ):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        if pad_token_id is None:
            pad_token_id = self.args.pad_token_id if self.args.pad_token_id is not None else 0
        attention_mask = self._prepare_attention_mask(attention_mask, idx)
        idx, attention_mask = self._left_pad_by_attention_mask(idx, attention_mask, pad_token_id)

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            mask_cond = None
            if attention_mask is not None:
                mask_cond = attention_mask if attention_mask.size(1) <= self.args.max_seq_len else attention_mask[:, -self.args.max_seq_len:]
            
            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond, attention_mask=mask_cond).logits
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出
            
            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            prev_finished = finished.clone()
            if stop_id is not None:
                if prev_finished.any():
                    fill_token = pad_token_id if pad_token_id is not None else stop_id
                    idx_next = torch.where(prev_finished[:, None], torch.full_like(idx_next, fill_token), idx_next)
                finished = prev_finished | idx_next[:, 0].eq(stop_id)

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)
            if attention_mask is not None:
                next_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                if prev_finished.any():
                    next_mask[prev_finished] = False
                attention_mask = torch.cat((attention_mask, next_mask), dim=1)

            if stop_id is not None and finished.all():
                break

        return idx[:, index:] # 只返回生成的token
    
    def _greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        贪婪解码：选择概率最大的token

        Args:
            logits: 模型输出的logits，形状为 (batch_size, vocab_size)

        Returns:
            选择的token索引，形状为 (batch_size, 1)
        """
        _, idx_next = torch.topk(logits, k=1, dim=-1)
        return idx_next

    def _random_sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        随机采样：基于概率分布随机选择token

        Args:
            logits: 模型输出的logits，形状为 (batch_size, vocab_size)
            temperature: 温度参数，控制随机性
            top_k: 只考虑概率最高的k个token

        Returns:
            选择的token索引，形状为 (batch_size, 1)
        """
        # 缩放 logits
        logits = logits / temperature

        # 应用top-k过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # 将不在 top-k 内的 logits 设为负无穷
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 计算概率并采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    def _beam_search(self, idx: torch.Tensor, max_new_tokens: int, num_beams: int,
                     temperature: float = 1.0, top_k: int = None, stop_id: int = None) -> torch.Tensor:
        """
        束搜索：维护多个候选序列，选择最优路径

        束搜索的核心思想：在每一步生成时，不是只选择一个最佳token，
        而是保留多个候选路径，最终选择累积概率最高的完整序列。

        Args:
            idx: 输入序列，形状为 (batch_size, seq_len)
            max_new_tokens: 最大生成token数量
            num_beams: 束宽度，表示保留的候选路径数量
            temperature: 温度参数，控制分布的平滑程度
            top_k: top-k过滤参数，限制候选token范围
            stop_id: 停止生成的token ID，遇到则停止

        Returns:
            生成的token序列，形状为 (batch_size, generated_length)
            只返回新生成的部分，不包含原始输入序列
        """
        # 获取输入序列的基本信息
        batch_size = idx.shape[0]  # 批次大小，通常为1
        seq_len = idx.shape[1]     # 输入序列长度

        # 初始化束：创建 num_beams 个候选序列
        beams = [idx.clone() for _ in range(num_beams)]
        # 初始化每个候选序列的累积对数概率分数
        beam_scores = torch.zeros(num_beams, device=idx.device)
        # 第一个候选是原始输入序列，分数为0
        beam_scores[0] = 0.0
        # 其他候选初始分数设为负无穷，表示尚未生成
        beam_scores[1:] = float('-inf')

        # 主循环：逐步生成新的token，最多生成 max_new_tokens 个
        for step in range(max_new_tokens):
            # 每轮迭代收集新的候选序列和分数
            new_beams = []   # 新的候选序列列表
            new_scores = []  # 对应的分数列表

            # 遍历当前的所有候选序列
            for beam_idx, beam in enumerate(beams):
                # 跳过无效候选（分数为负无穷的序列）
                if beam_scores[beam_idx] == float('-inf'):
                    continue

                # 序列长度检查：如果超过最大长度，截取最后的部分
                beam_cond = beam if beam.size(1) <= self.args.max_seq_len else beam[:, -self.args.max_seq_len:]

                # 前向传播：获取模型对当前序列的预测
                output = self(beam_cond)
                # 提取最后一个位置的logits，用于预测下一个token
                logits = output.logits[:, -1, :]  # 形状: (1, vocab_size)

                # 温度缩放：调整logits的分布
                if temperature != 1.0:
                    logits = logits / temperature
                    # 温度 > 1：分布更平滑，增加随机性
                    # 温度 < 1：分布更尖锐，更确定

                # Top-k过滤：限制候选token的范围，提高质量
                if top_k is not None:
                    # 找到logits中前top_k个最大的值
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # 将不在前top_k内的logits设为负无穷
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    # 这样采样时只会考虑前top_k个token

                # 计算对数概率：使用log_softmax避免数值不稳定
                log_probs = F.log_softmax(logits, dim=-1)

                # 获取前 num_beams 个最可能的候选token
                # 注意：这里的top-k与上面的top-k不同
                # 上面的top-k是全局过滤，这里是束搜索的分支选择
                top_log_probs, top_indices = torch.topk(log_probs, k=num_beams, dim=-1)

                # 为当前候选序列生成 num_beams 个扩展序列
                for k in range(num_beams):
                    # 选择第k个候选token
                    token = top_indices[:, k:k+1]      # token ID
                    log_prob = top_log_probs[:, k]     # 对应的对数概率

                    # 扩展序列：将新token添加到当前序列末尾
                    new_beam = torch.cat([beam, token], dim=1)
                    # 更新累积分数：原序列分数 + 新token的对数概率
                    new_score = beam_scores[beam_idx] + log_prob.item()

                    # 保存新的候选序列和分数
                    new_beams.append(new_beam)
                    new_scores.append(new_score)

            # 安全检查：如果没有生成任何有效候选，提前结束
            if not new_beams:
                break

            # 筛选最佳候选：从所有新生成的候选中选择分数最高的 num_beams 个
            # 按分数降序排序，获取索引
            sorted_indices = sorted(range(len(new_scores)), key=lambda i: new_scores[i], reverse=True)
            # 选择前 num_beams 个最佳候选
            beams = [new_beams[i] for i in sorted_indices[:num_beams]]
            beam_scores = [new_scores[i] for i in sorted_indices[:num_beams]]

            # 停止条件检查：检查最佳序列是否以停止token结尾
            if stop_id is not None and beams[0][0, -1] == stop_id:
                break

        # 返回得分最高的序列，只返回新生成的部分（去掉原始输入）
        # beams[0] 是最终得分最高的完整序列
        # [:, seq_len:] 切片只保留生成部分
        return beams[0][:, seq_len:]

    @torch.inference_mode()
    def generate_super(self,
                       idx,
                       stop_id=None,
                       max_new_tokens=256,
                       temperature=1.0,
                       top_k=None,
                       do_sample=False,
                       num_beams=1,
                       attention_mask: Optional[torch.Tensor] = None,
                       pad_token_id: Optional[int] = None
                       ):
        """
        高级文本生成函数，支持三种解码策略：

        1. 贪婪解码（Greedy Search）：
           - 参数：do_sample=False, num_beams=1
           - 特点：每步选择概率最大的token，速度快、结果确定

        2. 随机采样（Random Sampling）：
           - 参数：do_sample=True, num_beams=1
           - 特点：基于概率分布随机采样，可配合temperature和top-k控制多样性

        3. 束搜索（Beam Search）：
           - 参数：do_sample=False, num_beams>1
           - 特点：维护多条候选路径，选择总概率最高的序列，质量更高但速度较慢

        Args:
            idx: 输入序列张量，形状为 (batch_size, seq_len)
            stop_id: 停止生成的token ID
            max_new_tokens: 最大生成token数量
            temperature: 温度参数，控制随机性，越高越随机
            top_k: 只考虑概率最高的k个token，None表示不考虑
            do_sample: 是否使用随机采样，False时使用确定性解码
            num_beams: 束搜索的束宽度，1表示不使用束搜索

        Returns:
            生成的token序列，形状为 (batch_size, generated_length)
        """
        # 参数验证
        if temperature <= 0:
            temperature = 0.001  # 避免除零错误
        if num_beams < 1:
            num_beams = 1
        if top_k is not None and top_k < 1:
            top_k = None
        if pad_token_id is None:
            pad_token_id = self.args.pad_token_id if self.args.pad_token_id is not None else 0
        attention_mask = self._prepare_attention_mask(attention_mask, idx)
        idx, attention_mask = self._left_pad_by_attention_mask(idx, attention_mask, pad_token_id)

        # 束搜索逻辑
        if not do_sample and num_beams > 1:
            return self._beam_search(idx, max_new_tokens, num_beams, temperature, top_k, stop_id)

        # 贪婪解码和随机采样逻辑
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            mask_cond = None
            if attention_mask is not None:
                mask_cond = attention_mask if attention_mask.size(1) <= self.args.max_seq_len else attention_mask[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond, attention_mask=mask_cond).logits
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出

            # 根据参数选择解码策略
            if do_sample:
                idx_next = self._random_sample(logits, temperature, top_k)
            else:
                # 当temperature=0时使用贪婪解码
                if temperature < 0.1:
                    idx_next = self._greedy_decode(logits)
                else:
                    # 低温度下的随机采样（接近贪婪）
                    idx_next = self._random_sample(logits, temperature, top_k)

            prev_finished = finished.clone()
            if stop_id is not None:
                if prev_finished.any():
                    fill_token = pad_token_id if pad_token_id is not None else stop_id
                    idx_next = torch.where(prev_finished[:, None], torch.full_like(idx_next, fill_token), idx_next)
                finished = prev_finished | idx_next[:, 0].eq(stop_id)

            # 将选择的token添加到序列中
            idx = torch.cat((idx, idx_next), dim=1)
            if attention_mask is not None:
                next_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                if prev_finished.any():
                    next_mask[prev_finished] = False
                attention_mask = torch.cat((attention_mask, next_mask), dim=1)

            if stop_id is not None and finished.all():
                break

        return idx[:, index:] # 只返回生成的token

