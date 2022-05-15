# 使用CNN(+SPP結構)分辨葡萄藤葉子圖片判斷相關疾病

資料來源：kaggle - Grapevine Disease Images

https://www.kaggle.com/datasets/piyushmishra1999/plantvillage-grape

使用CNN結構分析葡萄藤葉子圖片，並將原本的Adaptive Pool替換成彈性SPP(Spatial Pyramid Pooling)結構。

使用kaggle中提供的所有圖片(共4組分類、4062張圖片)，按照6:2:2比例切分成訓練、驗證與測試資料集

結果：在第119epoch時三組資料集均達到99%以上準確率。
