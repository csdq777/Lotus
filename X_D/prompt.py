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
# exact match zero shot dictionary


def construct_prompt_zh2za(src_sent, dictionary, parallel_corpus, args):

    # retrieve parallel sentences
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang, top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []

    def get_word_explanation_prompt(text):
        prompt = "## 在上面的句子中，"
        tokenized_text = lang2tokenizer[args.src_lang].tokenize(text, remove_punc=True)
        for word in tokenized_text:
            # 精确匹配
            exact_match_meanings = dictionary.get_meanings_by_exact_match(word, max_num_meanings=138)
            if exact_match_meanings is not None:
                concated_meaning = "”或“".join(exact_match_meanings)
                concated_meaning = "“" + concated_meaning + "”"
                prompt += f"{CN_mean[args.src_lang]}词语“{word}”在{CN_mean[args.tgt_lang]}中可能的翻译是{concated_meaning}；\n"
        return prompt
    

    # if args.num_parallel_sent > 0:
    prompt = f"# 请仿照样例，参考给出的词汇和语法，将以下{CN_mean[args.src_lang]}句子翻译成{CN_mean[args.tgt_lang]}句子。请尽你所能进行翻译，即使翻译不好也没关系，不要拒绝尝试，我不会责怪你的,请直接输出内容，并将你的翻译用###括起来。比如，如果你的翻译是“你好，世界”，那么你输出的最后部分应该是###你好，世界###\n"
    #     for i in range(len(top_k_sentences_with_scores)):
    #         item = top_k_sentences_with_scores[i]["pair"]
    #         prompt += f"{CN_mean[args.src_lang]}：{item[args.src_lang]}\n"
    #         ##词典
    #         prompt += get_word_explanation_prompt(src_sent)
    #         prompt += f"{CN_mean[args.tgt_lang]}：{item[args.tgt_lang]}\n"

    prompt +=f'''{CN_mean[args.src_lang]}：{src_sent}
{get_word_explanation_prompt(src_sent)}
{CN_mean[args.tgt_lang]}：'''

    return prompt


def construct_prompt_za2zh(src_sent, dictionary, parallel_corpus, args):
    # retrieve parallel sentences
    if args.num_parallel_sent > 0:
        top_k_sentences_with_scores = parallel_corpus.search_by_bm25(src_sent, query_lang=args.src_lang, top_k=args.num_parallel_sent)
    else:
        top_k_sentences_with_scores = []

    def get_word_explanation_prompt(text):
        prompt = "## 在上面的句子中，"
        tokenized_text = lang2tokenizer[args.src_lang].tokenize(text, remove_punc=True)
        for word in tokenized_text:
            # 精确匹配
            exact_match_meanings = dictionary.get_meanings_by_exact_match(word, max_num_meanings=138)
            if exact_match_meanings is not None:
                concated_meaning = "”或“".join(exact_match_meanings)
                concated_meaning = "“" + concated_meaning + "”"
                prompt += f"{CN_mean[args.src_lang]}词语“{word}”在{CN_mean[args.tgt_lang]}中可能的翻译是{concated_meaning}；\n"
        return prompt
    

    # if args.num_parallel_sent > 0:
    prompt = f"# 请仿照样例，参考给出的词汇和语法，将以下{CN_mean[args.src_lang]}句子翻译成{CN_mean[args.tgt_lang]}句子。请尽你所能进行翻译，即使翻译不好也没关系，不要拒绝尝试，我不会责怪你的,请直接输出内容，并将你的翻译用###括起来。比如，如果你的翻译是“你好，世界”，那么你输出的最后部分应该是###你好，世界###'''\n"
    #     for i in range(len(top_k_sentences_with_scores)):
    #         item = top_k_sentences_with_scores[i]["pair"]
    #         prompt += f"{CN_mean[args.src_lang]}：{item[args.src_lang]}\n"
    #         ##词典
    #         prompt += get_word_explanation_prompt(src_sent)
    #         prompt += f"{CN_mean[args.tgt_lang]}：{item[args.tgt_lang]}\n"

    prompt +=f'''{CN_mean[args.src_lang]}：{src_sent}
{get_word_explanation_prompt(src_sent)}
{CN_mean[args.tgt_lang]}：'''
    
    return prompt



if __name__ == '__main__':
    pass

    



