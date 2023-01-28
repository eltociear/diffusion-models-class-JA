# ユニット2：ファインチューニング、ガイダンス、コンディショニング

Hugging Face Diffusion モデルコースのユニット2へようこそ! このユニットでは、事前にトレーニングされた diffusion モデルを新しい方法で使用し、適応させる方法を学びます。また、生成プロセスを制御するために、**コンディショニング** として追加の入力を受ける diffusion モデルをどのように作成するかもご覧いただきます。

## このユニットを開始する :rocket:

ユニットの手順は以下の通りです:

- 新しい教材がリリースされたときに通知を受けることができるように、[このコースにサインアップ](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)していることを確認してください
- 以下の資料に目を通し、本ユニットの重要な考え方の概要を理解してください
- _**Fine-tuning and Guidance**_ ノートブックで、🤗 Diffusers ライブラリを使用して新しいデータセットで既存の拡散モデルを微調整し、ガイダンスを使用してサンプリング手順を変更することを確認します
- ノートブックの例に従って、カスタムモデルの Gradio デモを共有します
- (オプション) _**Class-conditioned Diffusion Model Example**_ ノートブックで、生成プロセスにどのような制御を加えることができるかを確認してください。
- (オプション) [このビデオ](https://www.youtube.com/watch?v=mY20iKOQ2zw) で、このユニットの教材を非公式に確認することができます。


:loudspeaker: [Discord](https://huggingface.co/join/discord) に参加するのを忘れないでください。ここでは、 `#diffusion-models-class` チャンネルで教材について議論したり、作ったものを共有したりすることができます。
## ファインチューニング

ユニット1でご覧になったように、 diffusion モデルをゼロからトレーニングするのは時間がかかるものです。特に高解像度になればなるほど、ゼロからモデルをトレーニングするために必要な時間とデータは現実的ではなくなります。幸いにも、解決策があります：すでにトレーニングされたモデルから始めるのです。この方法では、ある種の画像のノイズ除去をすでに学習したモデルから始めます。これは、ランダムに初期化されたモデルから始めるよりも良い出発点になることを期待しています。

![Example images generated with a model trained on LSUN Bedrooms and fine-tuned for 500 steps on WikiArt](https://api.wandb.ai/files/johnowhitaker/dm_finetune/2upaa341/media/images/Sample%20generations_501_d980e7fe082aec0dfc49.png)

ファインチューニングは通常、新しいデータがベースモデルの元の学習データにある程度似ている場合にうまくいきますが（例えば、アニメの顔を生成しようとする場合、顔で学習したモデルから始めるとよいでしょう）、驚くことに、ドメインが大幅に変更された場合でも、その効果は持続するのです。上の画像は、 [LSUN Bedrooms データセットで学習したモデル](https://huggingface.co/google/ddpm-bedroom-256)と [WikiArt データセット](https://huggingface.co/datasets/huggan/wikiart)で500ステップのファインチューニングを行ったものです。[学習スクリプト](https://github.com/huggingface/diffusion-models-class/blob/main/unit2/finetune_model.py)は、このユニットのノートブックと一緒に参考として添付されています。

## ガイダンス

無条件モデルは生成されるものをあまりコントロールできない。条件付きモデル（詳しくは次のセクションで説明します）をトレーニングして、追加の入力を受け取り、生成プロセスをコントロールすることはできますが、すでにトレーニングされた無条件モデルがある場合はどうでしょうか？ガイダンスとは、生成プロセスの各ステップにおけるモデルの予測値を何らかのガイダンス関数に照らして評価し、最終的に生成される画像がより私たちの好みに合うように修正するプロセスです。

![guidance example image](guidance_eg.png)

このガイダンス関数は、ほとんどどんなものでも可能であり、強力な技法となります。このノートでは、単純な例（上の出力例のように色を制御する）から、 CLIP と呼ばれる事前に学習させた強力なモデルを利用し、テキストの記述に基づいて生成を誘導する例まで構築しています。

## コンディショニング

ガイダンスは無条件 diffusion モデルからいくつかのマイルを得るための素晴らしい方法ですが、もし学習中に利用可能な追加情報（クラスラベルや画像のキャプションなど）があれば、それをモデルに与え、予測を行う際に利用することも可能です。そうすることで、 **条件付き** モデルを作成し、推論時に条件付けとして入力されるものを制御することができます。ノートブックには、クラスラベルに従って画像を生成することを学習するクラス条件付きモデルの例が示されています。

![conditioning example](conditional_digit_generation.png)

このコンディショニング情報を渡す方法は、次のようなものがあります
- UNet への入力に追加チャンネルとして送り込む。これは、条件付け情報が画像と同じ形である場合によく使われます。例えば、セグメンテーションマスク、深度マップ、画像のぼやけたバージョン（復元／超解像モデルの場合）などがそうである。他のタイプの条件付けにも有効です。例えば、ノートブックでは、クラスラベルを埋め込みにマッピングし、入力画像と同じ幅と高さになるように拡張して、追加チャンネルとして入力できるようにしています。
- エンベッディングを作成し、それを UNet の1つ以上の内部レイヤーの出力でチャンネル数に見合ったサイズに投影し、それらの出力に追加していく。例えば、タイムステップのコンディショニングはこのように処理されます。各 Resnet ブロックの出力には、投影されたタイムステップのエンベッディングが追加されます。これは、 CLIP 画像エンベッディングのようなベクトルをコンディショニング情報として持っている場合に有効です。注目すべき例は、まさにこれを行う ['Image Variations' version of Stable Diffusion](https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations) です。
- 条件付けとして渡されたシーケンスに 'attend' できるクロスアテンションレイヤーを追加する。テキストは変換モデルを用いてエンベッディングのシーケンスにマッピングされ、 UNet のクロスアテンションレイヤーはこの情報をノイズ除去パスに取り込むために使用されます。ユニット3では、 Stable Diffusion がどのようにテキストコンディショニングを扱うかを検証するため、これを実際に見ていきます。


## ハンズオンノートブック

| 章                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fine-tuning and Guidance                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              |
| Class-conditioned Diffusion Model Example                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              |

この時点で、付属のノートブックを使い始めるのに十分な知識があります。上記のリンクから好きなプラットフォームで開いてみてください。もし Kaggle や Google Colab を使っているなら、最良の結果を得るためにランタイムタイプを 'GPU' に設定することを確認してください。

教材の大部分は、 _**Fine-tuning and Guidance**_ にあり、実例を通してこれら2つのトピックを探求しています。このノートブックでは、新しいデータで既存のモデルを微調整し、ガイダンスを追加し、その結果を Gradio のデモとして共有する方法を示しています。付属のスクリプト( [finetune_model.py](https://github.com/huggingface/diffusion-models-class/blob/main/unit2/finetune_model.py) )は、様々な微調整の設定を簡単に試すことができ、 🤗 Spaces 上で自分のデモを共有するためのテンプレートとして使用できる [example space](https://huggingface.co/spaces/johnowhitaker/color-guided-wikiart-diffusion) が用意されています。

_**Class-conditioned Diffusion Model Example**_ では、 MNIST データセットを用いて、クラスラベルを条件とした拡散モデルを作成する簡単な作業例を示します。これは、モデルにノイズ除去のための情報を与えることで、推論時にどのような種類の画像が生成されるかを制御することができます。

## プロジェクトタイム

ノートブック _**Fine-tuning and Guidance**_ の例に従って、あなた自身のモデルを微調整するか、既存のモデルを選び、あなたの新しいガイダンススキルを紹介する Gradio デモを作成してください。デモを Discord や Twitter などで共有し、あなたの作品を賞賛することを忘れないでください！

## 追加資料

[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) - DDIM サンプリングメソッドを導入（ DDIMScheduler で使用）

[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741) - diffusion モデルをテキストに条件付けする手法を導入

[eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://arxiv.org/abs/2211.01324) - さまざまなコンディショニングを併用することで、生成されるサンプルをより自在にコントロールすることが可能です

もっと素晴らしいリソースがありますか？このリストに追加します。
