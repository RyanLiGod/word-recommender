# -*- coding: utf-8 -*-
import gensim
from mycut import FilterCut
from numpy import array
import hnswlib


class Convert2Vec(object):
    def __init__(self, wm):
        self.wm = wm

    def text2v(self, text, cuttor):
        tokens = cuttor.fltcut(text)
        if len(tokens) == 0:
            return None
        else:
            return self.tokens2v(tokens)

    def tokens2v(self, tokens):
        assert len(tokens) > 0
        vectors = [self.wm[w] for w in tokens if w in self.wm]
        if len(vectors) == 0:
            return [0.0 for i in range(self.wm.vector_size)]
        return array(vectors).mean(axis=0)


class WordRecommander(object):
    def __init__(self):
        print("Loading word2vec bin...")
        self.wm = gensim.models.KeyedVectors.load_word2vec_format("wm.bin", binary=True)
        print("Finished.")
        self.t2v = Convert2Vec(self.wm)
        self.cuttor = FilterCut()
        self.vec_data = []
        self.vec_label = []
        self.vec_label_int = []
        print("Preparing data...")
        with open('Techword.txt', 'r') as f:
            n = 0
            for line in f.readlines():
                vec = recmder.get_word_vec(line.strip())
                self.vec_data.append(vec)
                self.vec_label.append(line.strip())
                self.vec_label_int.append(n)
                n += 1

    def build_hnsw(self, vec_data, vec_label_int):
        print("Initing index...")
        self.p = hnswlib.Index(space='l2', dim=200)
        self.p.init_index(max_elements=1100000, ef_construction=400, M=32)
        self.p.set_ef(200)
        print("Building index...")
        self.p.add_items(vec_data, vec_label_int)
        print("Saving index to 'tech_word.ind'...")
        self.p.save_index("tech_word.ind")

    def load_hnsw(self):
        self.p = hnswlib.Index(space='l2', dim=200)
        print("Loading index from 'tech_word.ind'\n")
        self.p.load_index("tech_word.ind", max_elements=1100000)

    def get_similar_words(self, word):
        vec = self.get_word_vec(word)
        labels, _ = self.p.knn_query(vec, k=10)
        neighbors = []
        for l in labels:
            neighbors.append(array(self.vec_label)[l])
        return labels

    def get_word_vec(self, word):
        vec = self.t2v.text2v(word, self.cuttor)
        return vec


if __name__ == "__main__":
    recmder = WordRecommander()
    # 构建HNSW索引
    # recmder.build_hnsw(vec_data, vec_label_int)
    recmder.load_hnsw()
    neighbors = recmder.get_similar_words("区块链")
    print(neighbors)
    neighbors = recmder.get_similar_words("数据")
    print(neighbors)
