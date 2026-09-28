from collections import defaultdict


class BTE:
    """Byte Pair Encoding tokenizer."""

    END = '</w>'

    def __init__(self, corpus, num_merges=10):
        self.num_merges = num_merges
        self.merges = {}  # (pair) -> rank, in the order they were learned
        self.vocab = set()
        self.train(corpus)

    def _word_freqs(self, corpus):
        freqs = defaultdict(int)
        for word in corpus.split():
            symbols = tuple(word) + (self.END,)
            freqs[symbols] += 1
        return freqs

    def _get_stats(self, word_freqs):
        pairs = defaultdict(int)
        for symbols, freq in word_freqs.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, word_freqs):
        merged = pair[0] + pair[1]
        new_freqs = {}
        for symbols, freq in word_freqs.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_freqs[tuple(new_symbols)] = freq
        return new_freqs

    def train(self, corpus):
        word_freqs = self._word_freqs(corpus)
        for symbols in word_freqs:
            self.vocab.update(symbols)

        for rank in range(self.num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            word_freqs = self._merge_vocab(best, word_freqs)
            self.merges[best] = rank
            self.vocab.add(best[0] + best[1])

    def encode(self, word):
        symbols = list(word) + [self.END]
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if candidate is None:
                break
            merged = candidate[0] + candidate[1]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == candidate:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def decode(self, tokens):
        return ''.join(tokens).replace(self.END, ' ').strip()


if __name__ == '__main__':
    tokenizer = BTE('aabaacaad', num_merges=15)
    print('merges:', tokenizer.merges)
    print('encode("lower"):', tokenizer.encode('aabaacaadaa'))
    print('decode:', tokenizer.decode(tokenizer.encode('lower')))
