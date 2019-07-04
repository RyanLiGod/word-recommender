# word-recommender

相似科技词汇推荐系统。使用近似最近邻算法HNSW。

启动推荐服务：

``` bash
python vec.py
```
请求方式：`http://localhost:5000/similar?word=数据挖掘&k=10`
