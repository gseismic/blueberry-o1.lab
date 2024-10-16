import re
from collections import defaultdict

class BPE:
    def __init__(self, num_merges):
        """
        初始化 BPE 类。

        Args:
            num_merges (int): 要执行的合并次数。

        Example:
            >>> bpe = BPE(num_merges=10)
        """
        self.num_merges = num_merges
        self.bpe_codes = {}
        self.vocab = {}

    def _get_stats(self, vocab):
        """
        计算字典中每对字符的出现频率。

        Args:
            vocab (dict): 字典，键为带空格的单词，值为频率。

        Returns:
            dict: 每对字符及其出现频率的字典。

        Example:
            >>> vocab = {'l o w </w>': 2, 'l o w e r </w>': 1}
            >>> stats = bpe._get_stats(vocab)
            >>> print(stats)
            {('l', 'o'): 2, ('o', 'w'): 2, ('w', '</w>'): 2, ('l', 'o'): 1, ('o', 'w'): 1, ('w', 'e'): 1, ('e', 'r'): 1, ('r', '</w>'): 1}
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        """
        合并字典中出现的字符对。

        Args:
            pair (tuple): 需要合并的字符对。
            vocab (dict): 当前词汇表。

        Returns:
            dict: 合并后更新的词汇表。

        Example:
            >>> vocab = {'l o w </w>': 2, 'l o w e r </w>': 1}
            >>> new_vocab = bpe._merge_vocab(('l', 'o'), vocab)
            >>> print(new_vocab)
            {'lo w </w>': 2, 'lo w e r </w>': 1}
        """
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def fit(self, corpus):
        """
        训练 BPE 模型。

        Args:
            corpus (list of str): 文本语料，每行一个字符串。

        Example:
            >>> corpus = ["low low lower", "new new low", "newer newer low"]
            >>> bpe.fit(corpus)
        """
        # 初始化词汇表
        self.vocab = defaultdict(int)
        for line in corpus:
            for word in line.split():
                word = ' '.join(list(word)) + ' </w>'  # 添加结束符
                self.vocab[word] += 1

        for i in range(self.num_merges):
            pairs = self._get_stats(self.vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.vocab = self._merge_vocab(best_pair, self.vocab)
            self.bpe_codes[best_pair] = i

    def encode(self, word):
        """
        编码单个词。

        Args:
            word (str): 需要编码的单词。

        Returns:
            list: 编码后的单词列表。

        Example:
            >>> encoded_word = bpe.encode("newer")
            >>> print(encoded_word)
            ['n', 'ew', 'e', 'r', '</w>']
        """
        word = ' '.join(list(word)) + ' </w>'
        while True:
            pairs = self._get_stats({word: 1})
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if best_pair not in self.bpe_codes:
                break
            word = self._merge_vocab(best_pair, {word: 1}).keys()
            word = next(iter(word))  # 获取合并后的单词

        return word.split()


if __name__ == "__main__":
    # 示例文本语料
    corpus = [
        "low low lower",
        "new new low",
        "newer newer low",
    ]

    # 初始化 BPE，并训练
    bpe = BPE(num_merges=10)
    bpe.fit(corpus)

    # 编码示例词
    test_word = "newer"
    encoded_word = bpe.encode(test_word)
    print(f'Encoded "{test_word}": {encoded_word}')
    
    # 显示 BPE 码
    print("BPE Codes:")
    for pair, index in bpe.bpe_codes.items():
        print(f"{pair}: {index}")