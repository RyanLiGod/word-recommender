#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Wu Yuanchao <151050012@hdu.edu.cn>


import codecs
import jieba

class FilterCut():

    def __init__(self, user_dict_file="TechWord.txt"):
        self.userdict = user_dict_file
        print('正在加载自定义词典...')
        jieba.load_userdict(self.userdict)
        jieba.enable_parallel(8)
        print('正在建构词袋...')
        self.load_word_bag()

    def load_word_bag(self):
        ws = set()
        with codecs.open(self.userdict, 'r', 'utf-8') as df:
            for line in df:
                ws.add(line.strip().lower())
        self.wbag = frozenset(ws)

    def filter(self, tokens):
        return [t for t in tokens if t in self.wbag]

    def fltcut(self, text):
        return self.filter(jieba.cut(text))
