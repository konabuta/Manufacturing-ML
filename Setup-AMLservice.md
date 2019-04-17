# Azure Machine Learning service Workshop

## 1. Python 開発環境の準備
Azure Machine Learning service を利用するには、下記の準備が必要です。  
  
- __Python 開発環境の例__ 
    - [Azure Notebook(Preview)](https://notebooks.azure.com/)
    - [Visual Studio Code](https://code.visualstudio.com/)
    - [Databricks Notebook](https://azure.microsoft.com/ja-jp/services/databricks/)
    - [Jupyter Notebook](https://jupyter.org/)
  
ドキュメント ["Azure Machine Learning のための開発環境を構成する"](https://docs.microsoft.com/ja-jp/azure/machine-learning/service/how-to-configure-environment) を参考に開発環境をセットアップします。
  - Azure Notebook (Preview) は、Python SDK がプリインストールされているため手軽に始めることが可能

## 2. Azure Machine Learning service ワークスペース 作成
Azure Portal から Azure Machine Learning service ワークススペースを作成します。
- Azure Portal にログイン
- "Machine Learning service workspace" を検索
- 環境情報の入力
    - ワークスペース名 : ※ 例:azureml
    - subscription : ワークショップで使用するSubscriptionを指定
    - ResourceGroup : ※ 例:handson 
    - 場所：東南アジア or 米国東部 を推奨

    
- 手順詳細：["ワークスペースを管理する"](https://docs.microsoft.com/ja-jp/azure/machine-learning/service/how-to-manage-workspace) 
- 通常分析プロジェクトや組織グループ毎に割り当てます。  
  
  
  
## 3. コンテンツのインポート

本ワークショップのコンテンツは下記になります。
```
git clone https://github.com/konabuta/AzureMLlab.git
```
使用する環境に応じて、コンテンツをインポートします。  

### **Azure Notebook を利用する場合**
Azure Notebook から、GitHub にあるコンテンツのインポート
- Azure Notebookにログイン
- "Upload GitHub Repo"にGitHub RepoのURLを入力し、"Import"
- インポート完了まで数分要します。
- 手順詳細 ： [GitHub からプロジェクトをインポートする](https://docs.microsoft.com/ja-jp/azure/notebooks/create-clone-jupyter-notebooks#import-a-project-from-github)

### **ローカル環境 を利用する場合**

git clone などを用いてGitHubからクローンします。

```bash
# コマンド例
git clone https://github.com/konabuta/Manufacturing-ML.git
```

## 4. Azure ML service への接続とクラスター作成

ノートブック [初期設定.ipynb](./初期設定.ipynb) を開いて実行していきます。

最初に、Azure Machine Learning service ワークスペースへ接続をします。接続情報をConfigファイルに保存し、以降はそれを読み込むだけで接続情報をロードすることができます。  

Configファイルの作成
```python
ws = Workspace(
   subscription_id = "<サブスクリプションID>", 
   resource_group = "<リソースグループ名>", 
   workspace_name = "<ワークスペース名>"
)

# configファイルの作成
ws.write_config()
```

以降は下記コメントのみで接続可能
```python
ws = Workspace.from_config()
```

次に、GPU/CPUのPaaSのクラスター環境である Machine Learning Compute を作成します。

```python
compute_config = AmlCompute.provisioning_configuration(
    vm_size="Standard_NC6s_v3", ## GPU/CPUインスタンスの種類 
    min_nodes=1, # 最小ノード数
    max_nodes=1, # 最大ノード数
    vm_priority='lowpriority') ## lowpriority = 低優先 | dedicated = 専用
```
**無償サブスクリプションでは、`vm_size` に `STANDARD_D2_V2` などを指定して、`CPU` クラスターへ変更ください。**  


## サンプルコード

修正中...

## 参考

### - 仮想Python環境の構築とPython SDKインストール

- __仮想Python環境の作成__

```shell
# create a new Conda environment with Python 3.6, NumPy, and Cython
conda create -n myenv Python=3.6 cython numpy

# activate the Conda environment
conda activate myenv

# On macOS run
source activate myenv

```

- __Azure ML service Python SDK インストール__
```shell
pip install --upgrade azureml-sdk[notebooks,automl,explain] azureml-dataprep
```

### - 接続情報を直接記載する方法
```python
# 直接接続情報を記載
ws = Workspace.get(
    name='<ワークスペース名>',
    subscription_id='<サブスクリプションID>',
    resource_group='<リソースグループ名>'
                   )
```