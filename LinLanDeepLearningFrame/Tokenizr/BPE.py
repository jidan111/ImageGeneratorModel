from . import *
import re


class BPE(object):
    def __init__(self, target_size=100, max_epoch=100, add_char=False):
        self.vocab = defaultdict(int)
        self.target_size = target_size
        self.max_epoch = max_epoch
        self.add_char = add_char
        self.encode_embedding = {}
        self.decode_embedding = {}

    def set_vocab(self, text):
        text = text.strip().replace("\n", "").replace("\t", '').replace("\r", '')
        for i in text.split(" "):
            self.vocab[" ".join(list(i)) + r"</w>"] += 1

    def get_best_pair_length(self):
        tmp = defaultdict(int)
        max_num = -1
        best_pair = None
        for k, v in self.vocab.items():
            ks = k.split(' ')
            for i in range(len(ks) - 1):
                pair = (ks[i], ks[i + 1])
                if pair == (" ", "</w>"):
                    continue
                tmp[pair] += v
                if tmp[pair] > max_num:
                    best_pair = pair
                    max_num = tmp[pair]
        return best_pair

    def update_vocab(self, best_pair):
        old_s = " ".join(best_pair)
        new_vbocab = defaultdict(int)
        new_s = "".join(best_pair)
        for k, v in self.vocab.items():
            pattern = re.compile(r"(?<!\S)" + re.escape(old_s) + r"(?!\S)")
            k = pattern.sub(new_s, k)
            new_vbocab[k] = v
        self.vocab = new_vbocab

    def get_embedding_map(self):
        out = []
        for k, v in self.vocab.items():
            t = k.split(" ")
            for j in t:
                if len(j) == 1:
                    continue
                out.append(j)
        out = ["</null>", "</unk>", "</w>"] + list(set(out))
        if self.add_char:
            out = out + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~")
        for k, v in enumerate(out):
            self.encode_embedding[v] = k
            self.decode_embedding[k] = v

    def fit(self, text):
        self.set_vocab(text)
        length = sum(len(k.split(' ')) for k in self.vocab)
        best_pair = self.get_best_pair_length()
        n = 0
        while length > self.target_size and n < self.max_epoch:
            self.update_vocab(best_pair=best_pair)
            best_pair = self.get_best_pair_length()
            n += 1
            length -= 1
        self.get_embedding_map()

    def encode_word(self, x):
        assert len(self.encode_embedding) != 0, "编码本为空"
        out = []
        if "</w>" not in x:
            x = x + "</w>"
        left = 0
        length = len(x)
        while left < length:
            right = length
            while True:
                s = x[left:right]
                if s in self.encode_embedding:
                    out.append(self.encode_embedding[s])
                    left = right
                    break
                else:
                    if right == left + 1:
                        out.append(self.encode_embedding["</unk>"])
                        left += 1
                        break
                    else:
                        right -= 1
        return out

    def encode_sentence(self, txt):
        out = []
        for i in txt.split(" "):
            out += self.encode_word(i)
        return out

    def decode(self, arr):
        assert len(self.decode_embedding) != 0, "解码本为空"
        out = ""
        for i in arr:
            out += self.decode_embedding[i]
        return out.replace("</w>", " ")

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.encode_embedding, f)

    def load(self, file_name):
        with open(file_name, 'r') as f:
            self.encode_embedding = json.load(f)
        for k, v in self.encode_embedding.items():
            self.decode_embedding[v] = k


if __name__ == '__main__':
    with open("./data/english_word.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    model = BPE(add_char=False, max_epoch=400)
    model.fit(content)
    print(model.encode_embedding)
    x = """image_shape = (3, 64, 64)
    hidden_channels = 16
    params = (3, 8, 2)
    group_nums = 8
    t_dim = 128
    step = 1000
    schedule_name = "cosine"
    device = "cuda"
    attention=[True, False, False]"""
    encode = model.encode_sentence(x)
    print(encode)
    decode = model.decode(encode)
    print(decode)
