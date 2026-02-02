from corpus import lang2tokenizer
import random
import json

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}
CN_mean = {
    'zh':'汉语',
    'yi':'彝语',
    'mn':'蒙古语(传统蒙古语书写系统)'
}

# Zero shot
def construct_prompt_zh2za(src_sent, dictionary, parallel_corpus, args):

#     prompt = f'''{CN_mean[args.src_lang]}:{src_sent}
# {CN_mean[args.tgt_lang]}:(请只输入{CN_mean[args.tgt_lang]}句子，不要有其他任何无关输出)'''

    prompt = f'''# 请帮我把下面的句子从{CN_mean[args.src_lang]}翻译成{CN_mean[args.tgt_lang]}:
{src_sent}
请尽你所能进行翻译，即使翻译不好也没关系，不要拒绝尝试，我不会责怪你的。
请将你的翻译用###括起来。比如，如果你的翻译是“你好，世界”，那么你输出的最后部分应该是###你好，世界###'''
    return prompt


def construct_prompt_za2zh(src_sent, dictionary, parallel_corpus, args):
    
    prompt = f'''# 请帮我把下面的句子从{CN_mean[args.src_lang]}翻译成{CN_mean[args.tgt_lang]}:
{src_sent}
请尽你所能进行翻译，即使翻译不好也没关系，不要拒绝尝试，我不会责怪你的。
请将你的翻译用###括起来。比如，如果你的翻译是“你好，世界”，那么你输出的最后部分应该是###你好，世界###'''
    return prompt



if __name__ == '__main__':
    pass

    



