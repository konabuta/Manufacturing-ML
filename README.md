# Machine Learning for Manufacturing #

製造業における機械学習/深層学習のアナリティクスのユースーケースをまとめています。Azure Machine Learning などのPaaSサービスを利用しています。


## [品質管理 (Quality Control)](./Quality-Control) ##
### [**自動機械学習による品質予測モデル構築**](./Quality-Control/Quality-Prediction)
- 自動機械学習 Automated Machine Learning による品質予測モデル構築
- 工場の製造工程データのデモデータを利用

### [**モデル解釈手法による品質の要因探索**](./Quality-Control/Root-Cause-Analysis-Explainability)
- Azure Machine Learning Interpretability SDK によるモデル解釈
- SHAPによるモデル解釈

### [**決定木による品質の要因探索**](./Quality-Control/Statistics-approach)
- Scikit-Learn Decision Tree モデルによる品質不良の発生条件の理解

### 外観検査モデル構築とWindows MLへのデプロイ (作成中)
- Custom Vision Service による画像分類モデル構築
- ONNXモデルのWindows Machine Learning デプロイ

<br/>

## [設備保全 (Predictive Maintenance)](./Predictive-Maintenance) ##
### [**LSTMによる設備保全**](./Predictive-Maintenance/Predict-RUL-lstm-remote)
- LSTMによるRULの時系列予測モデル作成
- Machine Learning Compute による高速モデル学習

<br/>

## 需要予測 (Demand Forecasting) ##
### 自動機械学習による需要予測モデル構築 (作成中)

<br/>

## Azure 環境の準備

### Azure Machine Learning service 
Azure Machine Learning service は、機械学習/深層学習のプロセスを全てカバーする Pythonベースの PaaSサービスになります。データサイエンティスト・市民データサイエンティストが主な利用ユーザになります。

<img src="https://docs.microsoft.com/en-us/azure/machine-learning/service/media/overview-what-is-azure-ml/aml.png" width = "600">    
  
#### Python SDK Install
```
pip install --upgrade azureml-sdk[notebooks,automl,explain,contrib] azureml-dataprep
```

詳細は構築手順は[こちらのページ](./Setup-AMLservice.md)をご参照ください。