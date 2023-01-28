# DreamBooth ハッカソン 🏆

DreamBooth ハッカソンへようこそ! このイベントは、 Stable Diffusion モデルをあなたの画像でファインチューニングすることで、**パーソナライズするコミュニティイベントです**。そのためには、 [_DreamBooth_](https://arxiv.org/abs/2208.12242) という強力なテクニックを使います。このテクニックを使うと、モデルの出力領域に被験者（例えば、ペットやお気に入りの料理）を埋め込み、プロンプトに _ユニーク識別子_ をつけて合成することができるようになります。

このコンペティションは5つのテーマで構成されており、各テーマでは以下のカテゴリーに属するモデルを募集しています:

* **Animal 🐨:** このテーマを使って、あなたのペットやお気に入りの動物が、アクロポリスでくつろいだり、泳いだり、宇宙を飛んだりする画像を生成してください。
* **Science 🔬:** このテーマを使って、銀河、タンパク質、または自然科学や医学のあらゆる分野のクールな合成画像を生成してください。
* **Food 🍔:** このテーマを使って、あなたの好きな料理に Stable Diffusion をチューニングしてみてはいかがでしょうか。
* **Landscape 🏔:** このテーマを使って、お気に入りの山、湖、庭の美しい風景を生成してください。
* **Wildcard 🔥:** このテーマを使って、好きなカテゴリの Stable Diffusion モデルを作ってください。

**テーマごとに最も気に入った上位3モデルに賞品を贈呈**します。お好きなだけモデルを提出してください!

## はじめに

以下の手順で参加してください:

1. [Hugging Face Discord サーバ](https://huggingface.co/join/discord)に参加し、 `#dreambooth-hackathon` チャンネルでイベントの最新情報を確認してください。
2. 以下のリンクをクリックして、 [DreamBooth notebook](https://github.com/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb) を起動・実行し、モデルを学習させます。各プラットフォームで GPU ランタイムを選択し、モデルを高速に学習させるようにしてください。

| Notebook                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DreamBooth トレーニング                              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb)              |

**備考 👋:** DreamBooth Notebook では、ファインチューニングする Stable Diffusion モデルとして [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) のチェックポイントを使用します。ただ、適切なコンポーネントと安全性チェッカー（存在する場合）をロードするようにコードを調整する必要があります。微調整が必要な興味深いモデルには、以下のようなものがあります:

* [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
* [`prompthero/openjourney`](https://huggingface.co/prompthero/openjourney)
* [`stabilityai/stable-diffusion-2`](https://huggingface.co/stabilityai/stable-diffusion-2)
* [`hakurei/waifu-diffusion`](https://huggingface.co/hakurei/waifu-diffusion)
* [`stabilityai/stable-diffusion-2-1`](https://huggingface.co/stabilityai/stable-diffusion-2-1)
* [`nitrosocke/elden-ring-diffusion`](https://huggingface.co/nitrosocke/elden-ring-diffusion)

## 評価＆リーダーボード

モデルカードに `dreambooth-hackathon` タグを付けて、1つ以上の DreamBooth モデルを Hub にプッシュすると、賞品がもらえます（[例](https://huggingface.co/lewtun/ccorgi-dog/blob/main/README.md#L9))。これは [DreamBooth notebook](https://github.com/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb) で自動的に作成されますが、独自のスクリプトを実行している場合は追加する必要があります。

モデルはいいね！の数によって評価され、ハッカソンのリーダーボードで自分のモデルの順位を確認することができます:

* [DreamBooth リーダーボード](https://huggingface.co/spaces/dreambooth-hackathon/leaderboard)

## タイムライン

* **December 21, 2022** - 開始日
* **December 31, 2022** - Colab Pro の登録締切
* **January 22, 2023** - 最終投稿締切（リーダーボード締切）
* **January 23-27, 2023** - 各テーマの受賞者発表

特に断りのない限り、締め切りはすべて対応する日の午後11時59分（UTC）となります。

## 賞品

テーマごとに3つの賞があり、リーダーボードで**最もいいね！が多いモデルによって受賞者が決定されます**:

**1位受賞者**

* [Hugging Face Pro subscription](https://huggingface.co/pricing)1年分、または [Hugging Face merch store](https://store.huggingface.co/) の100ドル券をプレゼント

**2位受賞者**

* [_TransformersとのNLP_](https://transformersbook.com/)書籍、または [Hugging Face merch store](https://store.huggingface.co/) の50ドル券をプレゼント

**3位受賞者**

* [Hugging Face Pro subscription](https://huggingface.co/pricing)一ヶ月分、または [Hugging Face merch store](https://store.huggingface.co/) の15ドル券をプレゼント

また、ハッカソン 🔥に DreamBooth モデルを1台以上提出した参加者全員に、**修了証**を提供します。


## 計算

Google Colab は本イベントに協賛し、参加者100名（ランダムに選出）に料金 Colab Pro クレジットを提供します。クレジットの提供は2023年1月を予定しており、12月31日までにご登録をお願いします。お申し込みは、[こちらのフォーム](https://docs.google.com/forms/d/e/1FAIpQLSeE_js5bxq_a_nFTglbZbQqjd6KNDD9r4YRg42kDFGSb5aoYQ/viewform)からお願いします。

![](https://lh3.googleusercontent.com/-l6dUgmPOKMM/X7w3nNn3OpI/AAAAAAAALAg/74fTRiPqikMURTD_Dn4PzAVADey2_6lLwCNcBGAsYHQ/s400/colab-logo-128x128.png)

## FAQ

### ファインチューニングに使用できるデータは？

あなたが所有している画像や、寛容なライセンスで許可されている画像は、すべて使用することができます。顔に学習させたモデルを提出したい場合（例：ウィルカードの提出）、あなた自身の似顔絵を使用することをお勧めします。理想的には、できる限りあなた自身のデータを使用してください。あなたのペットやお気に入りの地元の景観の特徴をぜひ見てみたいですし、いいねや賞品は、個人的に素敵なことをした人に行く傾向があると思われます 😁。

### その他、テキスト反転のようなファインチューニングは可能ですか？

もちろんです このハッカソンは DreamBooth にフォーカスしていますが、他のファインチューニングテクニックを試すことも歓迎（推奨）されます。また、フレームワークやコード、サービスなど、コミュニティが喜ぶようなモデルを作るのに役立つものであれば、何でも使っていただいて構いません🥰。

