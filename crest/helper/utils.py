from tqdm import tqdm
import os
import numpy as np
import torch

#########################################
#############   NLP Utils ###############
#########################################
def is_whitespace(c, use_space=True):
    if (c == " " and use_space) or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def normalize(word, ignore_spaces=True):
    chars = []
    for c in word.lower():
        if c == ' ' and ignore_spaces:
            pass
        else:
            chars.append(c)
    return ''.join(chars)


def clean_str(str_in):
    str_out = ''
    for c in str_in:
        if not is_whitespace(c, use_space=False):
            str_out += c
    return str_out


def compact_text(paragraph_text):
    doc_tokens = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(' ')
                doc_tokens.append(c)
                prev_is_whitespace = False
            else:
                doc_tokens.append(c)
                prev_is_whitespace = False
    return ''.join(doc_tokens)


#########################################
###########   File I/O Utils ############
#########################################
def read_file(txtfile, join=False):
    with open(txtfile, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return (' '.join(content) if join else content)


def save2file(filename, listofstrings):
    # to read : self.word_vocab = f.read().split("\n")
    fp = open(filename, "w")
    for i in tqdm(range(len(listofstrings))):
        str_state = listofstrings[i] + '\n'
        fp.write(str_state)
    fp.close()


write_files = save2file


def print_bars(s, num=20):
    print("##" * int(num))
    print(s)
    print("##" * int(num))

#########################################
###########  DataStructure Utils ########
#########################################
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        item = self.items[0]
        del(self.items[0])
        return item

    def peek(self):
        return self.items[0]

    def size(self):
        return len(self.items)


#########################################
###########  PyTorch Utils   ############
#########################################
def to_tensor(x, cuda=True):
    t_x = torch.tensor(x)
    if cuda:
        t_x = t_x.cuda()
    return t_x


def to_numpy(x, cuda=True, var=False):
    if var:
        x = x.detach()
    if cuda:
        x = x.cpu()
    return x.numpy()


def make_one_hot(n, idx, make_torch=False):
    a = np.zeros((n,), dtype=np.float32)
    a[idx] = 1.
    return torch.tensor(a) if make_torch else a






