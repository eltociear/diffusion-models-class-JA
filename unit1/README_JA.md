# ユニット1：Diffusion モデル入門

Hugging Face Diffusion モデルコースのユニット1へようこそ!このユニットでは、拡散モデルの仕組みの基本を学び、
🤗 Diffusers ライブラリを使用して独自のモデルを作成する方法を学びます。

## このユニットを開始する :rocket:

ユニットの手順は以下の通りです:

- 新しい教材がリリースされたときに通知を受けることができるように、[このコースにサインアップ](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)していることを確認してください
- 以下の紹介資料と、興味のありそうな追加資料に目を通してください
- 以下の _**Introduction to Diffusers**_ ノートブックで、 🤗 Diffusers ライブラリを使った理論の実践をチェックしてみてください
- ノートブックまたはリンクされたトレーニングスクリプトを使用して、独自の diffusion モデルをトレーニングし、共有することができます
- (オプション) 最小限のゼロからの実装に興味があり、様々な設計上の決定を検討したい場合は、 _**Diffusion Models from Scratch**_ ノートブックでより深く掘り下げることができます
- (オプション) [このビデオ](https://www.youtube.com/watch?v=09o5cv6u76c)で、このユニットの教材をざっと見てみてください。


:loudspeaker: [Discord](https://huggingface.co/join/discord) に参加するのを忘れないでください。ここでは、 `#diffusion-models-class` チャンネルで教材について議論したり、作ったものを共有したりすることができます。

## Diffusion モデルとは何か？

Diffusion モデルとは、「生成モデル」と呼ばれるアルゴリズム群に比較的最近追加されたものです。生成モデリングの目的は、多くの学習例が与えられたときに、画像や音声などのデータを **生成する** ことを学習することです。優れた生成モデルは、学習データを正確にコピーすることなく、それに類似した **多様な** 出力セットを作成します。 diffusion モデルはどのようにしてこれを実現するのでしょうか？ここでは、説明のために画像生成のケースに焦点を当ててみましょう。

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> 図は DDPM 論文（https://arxiv.org/abs/2006.11239）より。 </em>
<p>

diffusion モデルの成功の秘密は、 diffusion プロセスの反復性にあります。生成はランダムなノイズから始まりますが、出力画像が現れるまで、何段階にもわたって徐々に洗練されていきます。各ステップにおいて、モデルは現在の入力から完全にノイズ除去されたバージョンまでどのように進むかを推定します。しかし、各ステップで小さな変更を加えるだけなので、初期段階（最終的な出力を予測することが非常に難しい段階）でのこの推定値の誤差は、後の更新で修正することができます。

モデルの学習は、他のタイプの生成モデルに比べて比較的簡単です。以下を繰り返します
1) 学習データから画像をいくつか読み込む
2) 様々な量のノイズを加える。このとき、極端にノイズの多い画像と完璧に近い画像の両方を「修正」（ノイズ除去）する方法をモデルがうまく推定できるようにすることが重要です。
3) ノイズがかかったバージョンの入力をモデルに送り込む
4) これらの入力に対して、モデルがどの程度ノイズ除去を行うかを評価する
5) この情報を使ってモデルの重みを更新する

学習されたモデルを使って新しい画像を生成するには、まず完全にランダムな入力から始めて、それを繰り返しモデルに与え、モデルの予測に基づいて毎回少しずつ更新していきます。後述するように、このプロセスを効率化し、できるだけ少ないステップで良い画像を生成できるようにするためのサンプリング手法が数多く存在します。

ユニット1では、これらの各ステップをハンズオンノートブックで詳しく紹介します。ユニット2では、このプロセスをどのように変更し、追加の条件付け（クラスラベルなど）やガイダンスなどの手法によって、モデルの出力にさらなる制御を加えることができるかを見ていきます。そしてユニット3と4では、安定した拡散と呼ばれる非常に強力な拡散モデルを探求します。このモデルは、テキストの説明文から画像を生成することができます。

## ハンズオンノートブック

この時点で、付属のノートブックに取り掛かるのに十分な知識があるのです!この2つのノートは、同じアイデアを異なる方法で表現しています。

| Chapter                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction to Diffusers                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)              |
| Diffusion Models from Scratch                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)              |

_**Introduction to Diffusers**_ では、 diffusers ライブラリのビルディングブロックを使用して、上記の様々なステップを紹介します。どのようなデータであっても、独自の拡散モデルを作成し、訓練し、サンプリングする方法をすぐに理解することができます。このノートブックの終わりには、サンプルの学習スクリプトを読んで修正し、拡散モデルを学習し、世界と共有することができるようになります。このノートブックはまた、このユニットに関連する主な演習を紹介します。ここでは、様々なスケールの拡散モデルのための良い'トレーニングレシピ'を一緒に考えようとします。

_**Diffusion Models from Scratch**_ では、同じステップ（データへのノイズの追加、モデルの作成、学習、サンプリング）を、 PyTorch でできるだけ簡単にゼロから実装したものを紹介します。そして、この'おもちゃの例'を diffusers のバージョンと比較し、両者の違いや改善された点を指摘します。ここでのゴールは、異なるコンポーネントとそこに込められた設計上の決定に慣れ、新しい実装を見るときに、重要なアイデアを素早く識別できるようにすることです。

## プロジェクトタイム

さて、基本を押さえたところで、1つまたは複数の拡散モデルをトレーニングしてみましょう!いくつかの提案は、 _**Introduction to Diffusers**_ のノートブックの最後に記載されています。あなたの結果、トレーニングレシピ、発見をコミュニティと共有し、これらのモデルをトレーニングする最良の方法を共同で見つけ出すことができるようにしてください。

## 追加資料

[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) は、 DDPM の背後にあるコードと理論の非常に詳細なウォークスルーで、すべての異なる構成要素を示す数学とコードです。
 数学とコードで、すべての異なる構成要素を示しています。また、多くの論文にリンクしているので、さらに詳しく読むことができます。

Hugging Face のドキュメント [Unconditional Image-Generation](https://huggingface.co/docs/diffusers/training/unconditional_training) に、公式のトレーニング例スクリプトを用いた diffusion モデルのトレーニング方法の例と、独自のデータセットを作成する方法を示すコードが掲載されています。

Diffusion Models に関する AI Coffee Break のビデオ: https://www.youtube.com/watch?v=344w5h24-h8

DDPM に関する Yannic Kilcher 氏のビデオ: https://www.youtube.com/watch?v=W-O7AZNzbzQ

もっと素晴らしいリソースがありますか？このリストに追加します。
