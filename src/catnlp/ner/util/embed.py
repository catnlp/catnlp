#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


def get_embed(embed_file, vocab, embed_dim=100, format="word2vec"):
    """
    获得词向量
    Args:
        embed_file(str): 词向量文件路径
        vocab(Vocab): 词典类
        embed_dim(int): 词向量维度
        format(str): 词向量文件格式
    Returns: 词向量
    """
    format = format.lower()
    if format == "word2vec":
        embed = get_word2vec(embed_file, vocab, embed_dim)
    else:
        raise ValueError(f"不支持文件格式：{format}")
    return embed


def get_word2vec(embed_file, vocab, embed_dim):
    """
    获得word2vec词向量
    Args:
        embed_file(str): 词向量文件路径
        vocab(Vocab): 词典类
        embed_dim(int): 词向量维度
    Returns: word2vec词向量
    """
    scale = np.sqrt(3.0 / embed_dim)
    word_size = vocab.get_word_size()
    word2vec = np.zeros([word_size, embed_dim])

    # 没有预置词表
    if not embed_file:
        return np.random.uniform(-scale, scale, [word_size, embed_dim])

    match = 0
    not_match = 0
    word2id = vocab.get_word2id()
    embed_dict = load_embed_file(embed_file, embed_dim)
    for word, idx in word2id.items():
        if word in embed_dict:
            word2vec[idx, :] = embed_dict[word]
            match += 1
        else:
            word2vec[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_match += 1
    print('match:%s, not_match:%s, oov%%:%s'
                 % (match, not_match, (not_match + 0.0) / word_size))

    return word2vec


def load_embed_file(embed_file, embed_dim):
    """
    加载词向量文件
    Args:
        embed_file(str): 词向量文件路径
        embed_dim(int): 词向量维度
    Returns: 词向量矩阵
    """
    embed_dict = dict()
    with open(embed_file, 'r', encoding='utf-8') as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            lens = len(tokens)
            embed = np.empty([1, embed_dim])
            embed[:] = tokens[lens - embed_dim:]
            word = " ".join(tokens[: lens - embed_dim])
            embed_dict[word] = embed
    return embed_dict
