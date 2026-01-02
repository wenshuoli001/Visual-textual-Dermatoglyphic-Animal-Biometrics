import gzip#导入了 gzip 模块，该模块提供了用于读写 gzip 文件的工具。
import html#导入了 html 模块，该模块提供了一些 HTML 相关的方法，如转义和反转义 HTML 字符串。
import os
from functools import lru_cache#从 functools 模块导入了 lru_cache 装饰器。lru_cache 可以缓存函数的结果，以便在后续调用中重用

import ftfy#导入了 ftfy 模块，该模块可以修复一些常见的文本编码问题。
import regex as re#regex 模块提供了正则表达式相关的工具。


@lru_cache()
def default_bpe():
    #定义了一个名为 default_bpe 的函数，该函数返回一个文件路径，该路径指向与当前脚本在同一目录下的 ../data/bpe_simple_vocab_16e6.txt.gz 文件。
    #这个函数使用了 lru_cache 装饰器，这意味着它的结果会被缓存，以便在后续调用中重用。
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():#定义了一个名为 bytes_to_unicode 的函数，该函数使用了 lru_cache 装饰器，这意味着它的结果会被缓存，以便在后续调用中重用。
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    这个函数用于创建一个字典，该字典的键是 UTF-8 字节，值是对应的 Unicode 字符串。
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))#这行代码创建了一个名为 bs 的列表，该列表包含了一系列 UTF-8 字节的值。
    cs = bs[:]#创建了一个名为 cs 的列表，该列表是 bs 的一个副本
    n = 0
    for b in range(2**8):#这段代码遍历所有可能的 8 位字节值（从 0 到 255），如果某个字节值不在 bs 列表中，就将其添加到 bs 列表中，并将对应的 Unicode 字符添加到 cs 列表中。
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]#这行代码将 cs 列表中的每个元素转换为对应的 Unicode 字符。
    return dict(zip(bs, cs))#返回一个字典，字典的键是 UTF-8 字节，值是对应的 Unicode 字符串


def get_pairs(word):#定义了一个名为 get_pairs 的函数，该函数接收一个参数 word。
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    这个函数用于获取 word 中的所有相邻符号对。
    """
    pairs = set()#创建了一个空集合 pairs，用于存储 word 中的所有相邻符号对。
    prev_char = word[0]#这行代码将 word 的第一个符号赋值给 prev_char。
    for char in word[1:]:#这行代码开始一个循环，遍历 word 中的每个符号，从第二个符号开始。
        pairs.add((prev_char, char))#将当前符号和前一个符号组成的对添加到 pairs 集合中。
        prev_char = char#将当前符号赋值给 prev_char，以便在下一次循环中使用。
    return pairs#


def basic_clean(text):#定义了一个名为 basic_clean 的函数，该函数接收一个字符串 text 作为输入。
    '''
    这个函数用于对 text 进行基本的清洗
    '''
    text = ftfy.fix_text(text)#使用 ftfy.fix_text 函数修复 text 中的编码问题
    text = html.unescape(html.unescape(text))#使用 html.unescape 函数两次反转义 text 中的 HTML 字符实体。
    return text.strip()#返回去除了前后空白字符的 text


def whitespace_clean(text):#定义了一个名为 whitespace_clean 的函数，该函数接收一个字符串 text 作为输入
    '''
    这个函数用于将 text 中的一个或多个连续空白字符替换为单个空格，并去除 text 的前后空白字符。
    '''
    text = re.sub(r'\s+', ' ', text)#使用 re.sub 函数将 text 中的一个或多个连续空白字符替换为单个空格。
    text = text.strip()#去除 text 的前后空白字符。
    return text#返回 text


class SimpleTokenizer(object):
    '''
    类用于实现一个简单的分词器。分词器的主要功能是将输入的文本数据分解成更小的单元（如单词或字符），以便进行进一步的处理，如分析、建模等。
    '''
    def __init__(self, bpe_path: str = default_bpe()):
        #定义了 SimpleTokenizer 类的初始化方法。这个方法接收一个参数 bpe_path，该参数指定了 BPE（Byte Pair Encoding）词汇表的路径。如果没有提供 bpe_path，则使用 default_bpe 函数返回的默认路径。
        self.byte_encoder = bytes_to_unicode()#创建了一个字节到 Unicode 的映射，该映射由 bytes_to_unicode 函数返回。
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}#创建了一个 Unicode 到字节的映射，该映射是 byte_encoder 的反转。
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')#读取并解压 bpe_path 指定的 gzip 文件，然后将文件内容解码为 UTF-8 字符串，并按行分割。
        merges = merges[1:49152-256-2+1]#选择 merges 列表中的一部分元素
        merges = [tuple(merge.split()) for merge in merges]#将 merges 列表中的每个元素分割为一个元组
        vocab = list(bytes_to_unicode().values())#创建了一个词汇表 vocab，该词汇表包含了所有的 Unicode 字符。
        vocab = vocab + [v+'</w>' for v in vocab]#将 vocab 列表中的每个元素添加一个 '</w>' 后缀，并将结果添加到 vocab 列表中。
        for merge in merges:#将 merges 列表中的每个元组连接为一个字符串，并将结果添加到 vocab 列表中。
            vocab.append(''.join(merge))
        
        vocab.pop(-1) # remove last one in vocab(jekyll) to keep vocab_size unchanged#移除了 vocab 列表中的最后一个元素
        vocab.extend(['<|mask|>', '<|startoftext|>', '<|endoftext|>']) # vocab_size 49408#将三个特殊符号添加到 vocab 列表中。
        # vocab.extend(['<|startoftext|>', '<|endoftext|>']) # vocab_size 49408
        self.encoder = dict(zip(vocab, range(len(vocab))))#创建了一个字典，该字典的键是 vocab 中的元素，值是对应的索引。
        self.decoder = {v: k for k, v in self.encoder.items()}#创建了一个字典，该字典的键是 encoder 的值，值是 encoder 的键。
        self.bpe_ranks = dict(zip(merges, range(len(merges))))#创建了一个字典，该字典的键是 merges 中的元素，值是对应的索引。
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|mask|>': '<|mask|>', '<|endoftext|>': '<|endoftext|>'}#创建了一个字典，该字典的键是三个特殊符号，值是对应的索引。
        self.pat = re.compile(r"""<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)#创建了一个正则表达式对象，该对象用于匹配文本中的单词、数字和其他字符。

    def bpe(self, token):#定义了 bpe 方法，该方法接收一个参数 token
        '''
        这段代码定义了 SimpleTokenizer 类的 bpe 方法，该方法实现了字节对编码（Byte Pair Encoding，BPE）算法，用于将输入的单词（token）分解为更小的单元。
        这个方法的输入是一个单词（token），输出是一个字符串，该字符串是由输入单词分解而成的。
        '''
        if token in self.cache:#检查 token 是否在缓存中，如果在，则直接返回缓存中的结果。
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)#将 token 转换为一个元组，并在最后一个字符后添加 '</w>' 后缀
        pairs = get_pairs(word)#获取 word 中的所有相邻字符对

        if not pairs:#检查 pairs 是否为空，如果为空，则返回 token 加上 '</w>' 后缀。
            return token+'</w>'

        while True:#开始一个无限循环，直到满足退出条件
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))#找出 pairs 中在 bpe_ranks 中的排名最小的字符对
            if bigram not in self.bpe_ranks:#检查 bigram 是否在 bpe_ranks 中，如果不在，则退出循环。
                break
            first, second = bigram#将 bigram 分解为两个字符 first 和 second
            new_word = []#初始化一个空列表 new_word，并开始一个循环，遍历 word 中的每个字符。初始化一个空列表 new_word，并开始一个循环，遍历 word 中的每个字符。
            i = 0
            while i < len(word):
                try:#尝试找出 word 中从 i 开始的第一个 first 的位置 j，并将 word[i:j] 添加到 new_word 中。如果找不到 first，则将 word[i:] 添加到 new_word 中，并退出循环
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    #检查 word[i] 是否等于 first，并且 word[i+1] 是否等于 second。如果是，则将 first+second 添加到 new_word 中，并将 i 加 2。否则，将 word[i] 添加到 new_word 中，并将 i 加 1。
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)#将 new_word 转换为元组，并赋值给 word
            word = new_word
            if len(word) == 1:#检查 word 的长度是否为 1，如果是，则退出循环。否则，获取 word 中的所有相邻字符对。
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)#将 word 中的所有字符用空格连接起来，并将结果赋值给 word
        self.cache[token] = word#将 word 添加到缓存中
        return word#返回 word

    def encode(self, text):#定义了 encode 方法，该方法接收一个字符串 text 作为输入。
        '''
        该方法将输入的文本清理、分词、编码为 UTF-8，然后进行 BPE 分词，并将每个 BPE 令牌转换为对应的整数。
        返回一个整数组成的列表，列表中的每个整数都是一个 BPE 令牌。
        '''
        bpe_tokens = []#初始化了一个空列表 bpe_tokens，用于存储 BPE 令牌
        text = whitespace_clean(basic_clean(text)).lower()#首先对 text 进行基本清理，然后清理空白字符，并将结果转换为小写。
        for token in re.findall(self.pat, text):#开始一个循环，遍历 text 中的每个符合 self.pat 模式的令牌。
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))#将 token 编码为 UTF-8，然后使用 byte_encoder 将每个字节转换为对应的 Unicode 字符。
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))#将 token 进行 BPE 分词，然后使用 encoder 将每个 BPE 令牌转换为对应的整数，并将结果添加到 bpe_tokens 列表中。
        return bpe_tokens#返回 bpe_tokens 列表

    def decode(self, tokens):#定义了 decode 方法，该方法接收一个由整数组成的列表 tokens 作为输入。
        '''
        该方法将输入的整数列表解码为 BPE 令牌，然后解码为 UTF-8 字符串，并将 '</w>' 替换为空格
        返回的字符串是原始文本。
        '''
        text = ''.join([self.decoder[token] for token in tokens])#使用 decoder 将 tokens 列表中的每个整数转换为对应的 BPE 令牌，并将结果连接为一个字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')#使用 byte_decoder 将 text 中的每个字符转换为对应的字节，然后解码为 UTF-8 字符串，并将 '</w>' 替换为空格
        return text#返回 text
