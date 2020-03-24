# Otto Group Product Classification Challenge

- Otto コンペで使用したコード．
- 基本的はstackingでスコアを上げました．

## コンペ概要

- 匿名化された93個の特徴量を用いて, ある商品がどのクラスに入るのかを予測する多クラス分類．
- 評価指標: logloss

## 結果

- NNs stacking + LightGBM stackingでprivate score `0.41169`(49th, 銀メダル相当)を記録．

  - 一層目にはLightGBM, NNs, ERT, K-NN _6(k = 32, 64, 90, 128, 512, 1024), 各9classのみへのLogistic Regression_ 9

- notebookは整理してまた後で載せます．
