🇯🇵 日本語 (現在表示中)  
🇰🇷 [한국어版を読む](README_kr.md)

---

# 🚗 Jetson Nano ベースの自動運転RCカー プロジェクト

> **Jetson Nano + PyTorch + OpenCV** を活用したライン追跡型の自動運転RCカー  
> リアルタイムのライン認識・操舵・速度制御まで自動化を実現

---

## 🧠 プロジェクト概要

Jetson Nano、カメラ、DC/サーボモーターを用いて道路ラインをリアルタイムで認識し、  
CNNベースのディープラーニングモデル（PilotNet）により操舵角を予測し、自動走行を行うプロジェクトです。

---

## 🎬 実行動画（デモ）

![デモGIF](./jetson_rc_car_demo.gif)  
> Jetson Nano によるリアルタイム推論と自動操縦の成功シーン

🔗 [完全版動画を見る（mp4）](https://github.com/mj99k14/self-driving-rc-car/blob/main/KakaoTalk_20241217_204918887.mp4)

---

## 🛠 使用技術

| カテゴリ | ツール |
|----------|--------|
| プログラミング言語 | Python 3.8 |
| ディープラーニング | PyTorch + PilotNet |
| 映像処理 | OpenCV 4.10.0 |
| ハードウェア | Jetson Nano, DCモーター, サーボモーター, L298N |
| OS/環境 | Ubuntu 18.04 + JetPack 4.6.1, CUDA 10.2 |

---

## 💡 主な実装機能

- 📷 **ライン認識**：OpenCV + CNN による道路中心線の検出  
- 🕹 **操舵・速度制御**：PWM制御によるモーター制御  
- 🔀 **マルチスレッド処理**：映像分析と制御を並列処理  
- 🧠 **自作学習モデル**：13,000枚以上のデータを学習しPilotNetを最適化  

---

## 📈 成果と性能

- 最終 Validation Loss：**0.2107**
- テスト環境における走行成功率：**78.8%**
- 曲線区間での認識率を向上させるため、データ拡張とパラメータ調整を実施済み

---

## 📂 プロジェクト構成

```
Autonomous-Vehicle-Project/
├── model/             # 学習済みモデル
├── data/              # ライン画像と操舵角ラベル
├── src/               # カメラ、制御、推論コード
├── utils/             # 補助モジュール
└── README.md
```

---

## 📎 関連資料

- 📄 [発表用PPTを見る](https://github.com/mj99k14/self-driving-rc-car/blob/main/잿슨나노제출.pptx)
- 💻 [GitHubリポジトリ](https://github.com/mj99k14/self-driving-rc-car)

---