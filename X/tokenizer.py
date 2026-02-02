import re
import jieba
import jieba.posseg as pseg
import regex

class Tokenizer():
    def __init__(self):
        pass

    def tokenize(self, text, remove_punc=False):
        text = text.lower()    
        if remove_punc:
            # 去除中文标点符号
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " ")
            # 去除英文标点符号
            for punc in ",.;?!":
                text = text.replace(punc, " ")
            # 去掉数字
            text = re.sub(r'\d+', '', text)
        else:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " " + punc + " ")
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")
        # 替换单引号
        text = text.replace("‘", "'").replace("’", "'")
        # 按空格分词
        tokenized_text = text.split(" ")
        # 去除空格
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text

class EngTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()


class KgvTokenizer(Tokenizer):
    def __init__(self):
        super().__init__() 

class ZaTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text, remove_punc=False):
        text = text.lower()    
        if remove_punc:
            # 去除中文标点符号
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " ")
            # 去除英文标点符号
            for punc in ",.;?!":
                text = text.replace(punc, " ")
            # 去掉数字
            text = re.sub(r'\d+', '', text)
        else:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " " + punc + " ")
            # 去除英文标点符号
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")
        # 替换单引号
        text = text.replace("‘", "'").replace("’", "'")
        # 按空格分词
        tokenized_text = text.split(" ")
        # 去除空格
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text


class ZhTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text, remove_punc=False, do_cut_all=False, cut_for_search=False):
        # 使用jieba分词
        text = text.lower()
        if remove_punc:
            # 去除中文标点符号
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, "")
            # 去掉数字
            text = re.sub(r'\d+', '', text)
        
        if cut_for_search:
            tokenized_text = jieba.lcut_for_search(text)
        else:
            tokenized_text = jieba.lcut(text, cut_all=do_cut_all)
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text
    
class YiTokenizer(Tokenizer):
    def __init__(self, dictionary_path='../data/yi_token_dict.txt'):
        super().__init__()
        self.dictionary = self.load_dictionary(dictionary_path)

    def load_dictionary(self, path):
        dictionary = set()
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip()
                if word:
                    dictionary.add(word)
        return dictionary

    def tokenize(self, text, remove_punc=False):
        text = text.lower()
        
        if remove_punc:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " ")
            for punc in ",.;?!":
                text = text.replace(punc, " ")
            text = re.sub(r'\d+', '', text)
        else:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " " + punc + " ")
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")

        # 双向匹配
        return self.bi_directional_match(text)

    def forward_match(self, text):
        tokenized_text = []
        idx = 0
        while idx < len(text):
            matched = False
            for end in range(len(text), idx, -1):
                word = text[idx:end]
                if word in self.dictionary:
                    tokenized_text.append(word)
                    idx = end
                    matched = True
                    break
            if not matched:
                tokenized_text.append(text[idx])
                idx += 1
        return tokenized_text

    def reverse_match(self, text):
        tokenized_text = []
        idx = len(text)
        while idx > 0:
            matched = False
            for start in range(0, idx):
                word = text[start:idx]
                if word in self.dictionary:
                    tokenized_text.insert(0, word)
                    idx = start
                    matched = True
                    break
            if not matched:
                tokenized_text.insert(0, text[idx - 1])
                idx -= 1
        return tokenized_text

    def bi_directional_match(self, text):
        forward_tokens = self.forward_match(text)
        reverse_tokens = self.reverse_match(text)

        # 合并并去重，优先保留正向结果
        combined_tokens = []
        seen = set()

        for token in forward_tokens:
            if token not in seen:
                combined_tokens.append(token)
                seen.add(token)

        for token in reverse_tokens:
            if token not in seen:
                combined_tokens.append(token)

        return combined_tokens

class MnTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text, remove_punc=False):
        # 定义正则表达式来匹配蒙古文字符
        word_pattern = regex.compile(
            "[\u180E\u202F]?[\u200D\u180A]*[\u1820-\u1842]"
            "(?:" 
            "[\u180A-\u180D\u180F\u200D\u1820-\u1842]"
            "|"
            "(?<=[\u180A\u200D\u1820-\u1842][\u180B\u180C\u180D\u180F]*"
            "(?:\u1828[\u180B\u180C]?|\u182C\u180B?|\u182D[\u180B\u180D]?|\u1830\u180B?|[\u182E\u182F\u1831\u1835-\u1838]))"
            "\u180E"
            "(?=[\u1820\u1821](?:[^\u180A-\u180D\u180F\u200D\u1820-\u1842]|$))"
            ")"
            "*"
        )

        # 将文本转换为小写
        text = text.lower()

        # 去除标点符号
        if remove_punc:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " ")
            for punc in ",.;?!":
                text = text.replace(punc, " ")
            text = re.sub(r'\d+', '', text) 
        else:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, " " + punc + " ")
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")

        # 替换单引号
        text = text.replace("‘", "'").replace("’", "'")

        # 使用正则表达式匹配蒙古文字符
        tokenized_text = word_pattern.findall(text)

        # 去除空格和空字符
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]

        return tokenized_text