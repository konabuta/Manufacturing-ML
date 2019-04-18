# Machine Learning for Manufacturing #

# 品質管理 (Quality Control) ##
## 自動機械学習による品質予測モデル構築
- 自動機械学習 Automated Machine Learning による品質予測モデル構築
- 工場の製造工程データのデモデータを利用

## モデル解釈手法による品質の要因探索
- SHAPを用いた変数の重要度の出力
- SHAPを用いた行レベルの変数寄与度の出力

## 決定木による品質の要因探索
- Scikit-Learn Decision Tree モデルによる品質不良の発生条件の理解

## 外観検査モデル構築とWindows MLへのデプロイ
- Custom Vision Service による画像分類モデル構築
- ONNXモデルのWindows Machine Learning デプロイ

# 設備保全 (Predictive Maintenance) ##
## LSTMによる設備保全

# 需要予測 (Demand Forecasting) ##
## 自動機械学習による需要予測モデル構築


# Azure 環境の準備

Azure Machine Learnaing service Python SDK

```
pip install --upgrade azureml-sdk[notebooks,automl,explain,contrib] azureml-dataprep
```

詳細は構築手順は[こちらのページ](./Setup-AMLservice.md)をご参照ください。
