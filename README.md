# metric_learning
metric_learningの練習用リポジトリ。今回はL2 softmax lossの実験。

## 使用フレームワーク
- PyTorch

## 実行環境
- Google colab

## 参考サイトおよびGithub
- https://qiita.com/daisukelab/items/d7ebfe8d7ade18def967
- https://github.com/daisukelab/general-learning/blob/master/Metric%20learning%20MNIST%20embeddings%20comparison.ipynb


## 実験のまとめ(自身の備忘録)
(現在整備中)

## code
以下、コードの説明です。

### main.py
- 生のデータを可視化

        python main.py 0

- ResNet18を経由したシンプルなSoftmaxモデル

        python main.py 1

- L2 Softmaxを用いたモデル(CNNモデルはResNet18)

        python main.py 2

### utils.py

データセットロードやモデルに関する追加修正


### visualize_data.py

t-SNEを用いた可視化用プログラム。