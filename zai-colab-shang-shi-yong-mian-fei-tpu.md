---
description: 免費仔有福了
---

# 在 Google Colab 上使用免費 TPU

Colab 上現在可以使用免費 TPU 了（雖然不是真的完全免費）

> 數據儲存費用是 USD 0.02/ GB。擔心花太多錢的話可以看詳細的[價目表](https://cloud.google.com/storage/pricing?hl=zh-tw)。另外[這裡](https://cloud.google.com/free?hl=zh-TW)可以申請新客戶免費 USD 300 ，可以用超久

對想自己練習建模或做 side-project 的人來說，簡直是天上掉下來的午餐

對免費仔來說，TPU 真的快、超快，甚至可能比很多學校實驗室提供的 GPU 還好用，但是寫法不太直覺。我在用 TPU 的過程中踩了很多坑，而且發現網路上很難找到完整的入門教學文章。我憑著極有限的技術能力一路拼拼湊湊、跌跌撞撞，這篇文章就是想記錄這些內容，讓大家建立起一個能動的模型（能動最重要了，其它再說吧）

**關於 Colab**

Google 提供的免費類 Jupyter Notebook 介面，付費版本可以優先使用比較快的 GPU, 並且提供更長的運行時長。不過如果只是想做 POC 或練習訓練一些小模型的人，用免費的就好了。但我必須要說 Colab Pro 真的很好用，可以提升訓練時長到 12 小時左右，而且 GPU 也快不少，而且一個月才 USD 9.99。比較麻煩的是需要填一個美國地址，我是找朋友幫掛了一個地址。（最近又出了 Colab Pro+，這邊有[定價](https://colab.research.google.com/signup)）

**關於 TPU**

Google 架設的運算資源集叢，真的很快，爬一些文章看大家的心得發現比單核 GPU 快 3 - 9 倍不等。我是使用 Colab Pro \(USD 9.99/月\), 已經有比較快的 GPU了，但是 TPU 跑起來大概快個 5 倍，讓我原本訓練一個模型要大概 5 天（而且會一直斷線XD），現在只需要一個晚上，真的大碗又滿意

**前置作業— 把資料放上雲端**

作為 Google Cloud 生態系的一部分，TPU 大部分應該是企業用戶在用。現在開放比較舊的 TPU 版本給 Colab 使用，但是在開始訓練之前，資料要全部放在 Google Cloud 的 GCS \(Google Cloud Storage\) 中，而把資料放在這上面需要花一點點錢。要注意的是這邊是不能用 Google Drive，一定要用 GCS

但是收費很低，不太需要擔心 \(儲存費用是 USD 0.026/ GB\)。如果資料量真的很大擔心花太多錢的話可以看詳細的[價目表](https://cloud.google.com/storage/pricing?hl=zh-tw)

如果你以前沒有用過 Google Cloud, 可以在[這裡](https://cloud.google.com/free?hl=zh-TW)申請新客戶免費額度，這樣就有 USD 300 可以用了，可以用超久

1. 首先在 Google Cloud Console 打開一個新 project

![](https://cdn-images-1.medium.com/max/800/1*3Wo2CGfwP5fWi1_70qaDTw.png)

2. 在新 project頁面搜尋 Google Cloud Storage

![](https://cdn-images-1.medium.com/max/800/1*TgGbHhVFX_7NPsYa-zlX-Q.png)

3. GCS 上的基本單位是 Bucket, 所以先創建一個新的 Bucket，取好名字後用預設選項一路選下去然後按 create 就可以了。右邊可以估算花費

![](https://cdn-images-1.medium.com/max/800/1*OjCO2jZCOOEpVLxRL48zcA.png)

創建好之後會進入 Bucket 頁面，把要用的資料都上傳到這邊。我的做法是直接把已經分好 train-val-test 的資料夾上傳

**Google Colab**

好了之後就可以進到 Colab 裡面了，在「執行階段」裡的「變更執行階段類型」選擇

![](https://cdn-images-1.medium.com/max/800/1*mAwqbRrSUOK7jzAeFdbt7A.png)

**1. 確認版本 \(Tensorflow\)**

這邊有個坑，Colab 上的 Tensorflow 版本要跟 TPU 上的 Tensorflow 版本一樣。但問題是我怎麼會知道 TPU 上的 Tensorflow 是什麼版本？總之，在這篇文章產出的當下 \(2021/8/20\) 是 2.5，而 Colab 預裝的 Tensorflow 也是 2.5, 所以不需要做任何事

但是在幾個月前，TPU的版本是 2.4，所以就要先執行

```text
!pip install tensorflow==2.4
```

如果遇到問題，真的找不出 bug 的話，可以試著更換 Tensorflow 版本

至於 PyTorch 的話…我就不知道了，大家可以看看這個針對 PyTorch [官方教程](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb)

**2. 權限設定**

首先要先讓 Colab 可以存取 GCS。另外，推薦將訓練好的模型存在 Google Drive，所以可以同時提供 GCS 和 Google Drive 的權限

要留意的是，Colab 的授權方式是，運行這一格之後，會出現一個授權頁面，裡面會有一串代碼。把這串代碼貼到 Colab output 上之後就算授權完成。所以**這兩個授權要寫在分開的兩格**，不然在 output 就只會出現第二個授權

**Google Drive 授權**

```text
# Access Google Drive
from google.colab import drive
drive.mount(‘/content/drive’)
```

**GCS 授權**

```text
# Access GCS
from google.colab import auth
auth.authenticate_user()
```

**3. 分散式運算咒語**

這一串是讓 Tensorflow 使用分散式運算。 3 和 4 都是參照[官方教程](https://colab.research.google.com/notebooks/tpu.ipynb)

```text
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    print(‘Running on TPU ‘, tpu.cluster_spec().as_dict()[‘worker’])
    
except ValueError:
    raise BaseException(‘ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!’)

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
```

**4. 設定 Batch Size 和路徑**

根據 TPU 文件，Batch Size 建議為 64 的倍數，可以從比較大的 Batch Size \(1024之類的\) 開始試。另外，因為 TPU 算力很強，如果餵資料不夠快，運算瓶頸就會在 data pipeline 那邊，所以一次餵多一點資料也是想充分運用 TPU

```python
batch_size = 1024
gcs_path = “gs://new-bucket/” # 在 GCS 上的 bucket name 
```

```text
train_pattern = gcs_path + “train/*.tfrecords”
train_files = tf.io.gfile.glob(train_pattern)
```

```text
val_pattern = gcs_path + “val/*.tfrecords”
val_files = tf.io.gfile.glob(val_pattern)
```

這邊跑完之後如果不放心可以把 train\_files 印出來檢查

**5. 資料和初始化模型**

大致上就跟平常的做法一樣，但使用 TPU 時要把這些東西都包成函數，所以我的做法就只是這樣：

```python
def train_data():
    # 就是普通的 training data
    return train
```

```python
def create_model():
    # 就是普通的 keras model
    return model
```

另外在建立模型時有一點要注意，有時候大家在指定維度的時候會習慣使用 -1 來代表最後一個維度，或是在 reshape 的時候會把最後一個數字寫成 -1. 在 TPU 架構上這是不行的，一定要把數字明確地寫出來

**6. 開始訓練**

```python
with tpu_strategy.scope(): 
    m = create_model()
    m.save("/content/drive/MyDrive/") # 儲存模型架構
    
m.fit(train_data(), ...)
```

最後訓練好的參數可以儲存在 Google Drive 上

跟著這些步驟的話，應該就可以順利用上 TPU 了，不過當然還是要附上：

**7. 如何 debug**

用 TPU 會遇到超多 bug, 所以這邊提供一個 debug 的清單，可以先從這個清單上開始檢查

* Google Cloud Storage 有沒有授權成功
* 餵資料和模型有沒有包成函數
* 模型裡的維度有沒有被「明確指出」 \(不可以寫 -1\)
* 初始化模型有沒有在 tpu\_strategy.scope\(\) 底下
* 使用 Tensorflow 的話，**不需要**特別指定 Eager / Graph mode
* 最好的 debug 手冊：TPU [官方文件](https://cloud.google.com/tpu/docs/tpus)
* TPU in Colab [官方範例](https://colab.research.google.com/notebooks/tpu.ipynb?hl=es) → 推薦大家可以從這個範例開始改 code

如果你已經有一個能動的模型了，可以試試看進階的 [Performance Profiling](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/profiling_tpus_in_colab.ipynb)

透過 Tensorboard 可以看到模型訓練的瓶頸點在哪，由於 TPU 運算能力很強，所以大部分的瓶頸都會是在餵資料那一端，在 Tensorboard 上可以看到 TPU 有多長時間是閒置的。如果瓶頸是在餵資料那邊，就要想辦法優化 data pipeline

祝大家善用快速又超低價的 TPU 資源，擺脫訓練的等待時間

