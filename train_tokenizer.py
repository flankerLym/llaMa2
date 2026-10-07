"""只用训练集训练 ByteLevel BPE，避免验证/测试泄漏。"""
import argparse,json
from pathlib import Path
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
from transformers import PreTrainedTokenizerFast
SPECIAL=['<pad>','<unk>','<s>','</s>','<|im_start|>','<|im_end|>']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/pretrain_train.jsonl')
    p.add_argument('--out',default='tokenizer'); p.add_argument('--vocab-size',type=int,default=6144)
    a=p.parse_args()
    def texts():
        with open(a.data,encoding='utf-8') as f:
            for line in f: yield json.loads(line)['text']
    backend=Tokenizer(models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder=decoders.ByteLevel()
    backend.train_from_iterator(texts(),trainers.BpeTrainer(vocab_size=a.vocab_size,min_frequency=2,
        special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',unk_token='<unk>',
        bos_token='<s>',eos_token='<|im_end|>',additional_special_tokens=['</s>','<|im_start|>'])
    tok.chat_template="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    tok.save_pretrained(a.out)
    for text in ['你好，世界！','Hello world!','<|im_start|>assistant\n']:
        assert tok.decode(tok.encode(text,add_special_tokens=False),clean_up_tokenization_spaces=False)==text
    print('Tokenizer roundtrip OK，实际词表大小：',len(tok))
if __name__=='__main__': main()
