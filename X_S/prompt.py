from corpus import lang2tokenizer
import random
import json

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}
CN_mean = {
    'zh':'汉语',
    'yi':'彝语',
    'mn':'蒙古语'
}
# Few shot



def construct_prompt_zh2za(src_sent, dictionary, parallel_corpus, args):
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang, top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []
    

    if args.num_parallel_sent > 0:
        prompt = f"# 请仿照样例，将{CN_mean[args.src_lang]}句子翻译成{CN_mean[args.tgt_lang]}句子。请尽你所能进行翻译，即使翻译不好也没关系，不要拒绝尝试，我不会责怪你的。请将你的翻译用###括起来。比如，如果你的翻译是“你好，世界”，那么你输出的最后部分应该是###你好，世界###\n"
        for i in range(len(top_k_sentences_with_scores)):
            item = top_k_sentences_with_scores[i]["pair"]
            prompt += f"{CN_mean[args.src_lang]}：{item[args.src_lang]}\n"
            prompt += f"{CN_mean[args.tgt_lang]}：{item[args.tgt_lang]}\n"

    prompt +=f'''{CN_mean[args.src_lang]}：{item[args.src_lang]}
{CN_mean[args.tgt_lang]}：'''

    return prompt


def construct_prompt_za2zh(src_sent, dictionary, parallel_corpus, args):
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang, top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []
    

    if args.num_parallel_sent > 0:
        prompt = f"# 请仿照样例，将{CN_mean[args.src_lang]}句子翻译成{CN_mean[args.tgt_lang]}句子。请尽你所能进行翻译，即使翻译不好也没关系，不要拒绝尝试，我不会责怪你的。请将你的翻译用###括起来。比如，如果你的翻译是“你好，世界”，那么你输出的最后部分应该是###你好，世界###'''\n"
        for i in range(len(top_k_sentences_with_scores)):
            item = top_k_sentences_with_scores[i]["pair"]
            prompt += f"{CN_mean[args.src_lang]}：{item[args.src_lang]}\n"
            prompt += f"{CN_mean[args.tgt_lang]}：{item[args.tgt_lang]}\n"

    prompt +=f'''{CN_mean[args.src_lang]}：{item[args.src_lang]}
{CN_mean[args.tgt_lang]}：'''

    return prompt




if __name__ == '__main__':
    pass

    



