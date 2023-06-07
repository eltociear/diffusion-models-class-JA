# ユニット3: Stable Diffusion

Hugging Face Diffusion モデルコースのユニット3へようこそ！このユニットでは、Stable Diffusion (SD) と呼ばれる強力な diffusion モデルに出会い、それができることを探ります。

## このユニットを開始する :rocket:

ユニットの手順は以下の通りです:

- 新しい教材がリリースされたときに通知を受けることができるように、[このコースにサインアップ](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)していることを確認してください。
- 以下の資料に目を通し、本ユニットの重要な考え方の概要を理解してください
- [_**Stable Diffusion Introduction**_ notebook](#hands-on-notebook) では、SD を実際に使用するケースをご紹介しています
- [**hackathon** フォルダ](https://github.com/huggingface/diffusion-models-class/tree/main/hackathon)にある _**Dreambooth**_ ノートブックを使って、あなただけの Stable Diffusion モデルを微調整し、コミュニティで共有すれば、賞品や賞品を獲得するチャンスがあります
- (オプション) [_**Stable Diffusion Deep Dive video**_](https://www.youtube.com/watch?app=desktop&v=0_BBRNYInx8) と付属の [_**notebook**_](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb) で、さまざまなコンポーネントとその効果について、より深く掘り下げてみてください。この教材は、FastAI の新しいコース「['Stable Diffusion from the Foundations'](https://www.fast.ai/posts/part2-2022.html)」のために作成されました。最初の数レッスンはすでに公開されており、残りのレッスンは今後数ヶ月で公開予定です。このようなモデルを完全にゼロから作ることに興味がある人にとっては、このクラスの素晴らしい補足になるはずです。


:loudspeaker: [Discord](https://huggingface.co/join/discord) に参加するのを忘れないでください。ここでは、 `#diffusion-models-class` チャンネルで教材について議論したり、作ったものを共有したりすることができます。

## はじめに

![SD example images](sd_demo_images.jpg)<br>
_Stable Diffusion で生成した画像例_

Stable Diffusion は、強力なテキスト条件付き潜在的拡散モデルです。心配しないでください、この言葉についてはすぐに説明します テキストの記述から素晴らしい画像を作成するその能力は、インターネット上でセンセーションを巻き起こしました。このユニットでは、 SD がどのように機能するかを探り、他にどのようなトリックができるかを見ていくことにします。

## Latent Diffusion

画像のサイズが大きくなると、その画像を処理するために必要な計算能力も大きくなります。特に自己アテンションと呼ばれる操作では顕著で、入力数に対して二次関数的に操作量が増えていきます。128 px の正方形画像は64 px の正方形画像の4倍の画素数を持つため、自己アテンション層では16倍（つまり4<sup>2</sup>）のメモリと計算が必要です。これは、高解像度の画像を生成したい人にとっての問題になります！

![latent diffusion diagram](https://github.com/CompVis/latent-diffusion/raw/main/assets/modelfigure.png)<br>
_[Latent Diffusion 論文](http://arxiv.org/abs/2112.10752)からの図_

この問題を軽減するために、 VAE（Variational Auto-Encoder） と呼ばれる別のモデルを用いて、画像をより小さな空間次元に圧縮することができます。十分な学習データがあれば、 VAE は入力画像をより小さく表現することを学習し、この小さな **潜在的** 表現に基づいて画像を忠実に再構成できることが期待されます。 SD で使用されている VAE は、3チャンネルの画像を取り込み、各空間次元の縮小率を8とした4チャンネルの潜像表現を生成します。つまり、512 px の正方形の入力画像は、4 x 64 x 64の潜像に圧縮されることになります。

フル解像度の画像ではなく、この **潜在的な表現** に拡散プロセスを適用することで、より小さな画像を使用することで得られる多くの利点（メモリ使用量の削減、UNet に必要なレイヤーの減少、生成時間の短縮など）を得ることができ、最終結果を表示する準備ができたら高解像度画像にデコードして結果を戻すこともできます。この革新的な技術により、モデルの訓練と実行にかかるコストが劇的に削減されます。

## テキストコンディショニング

ユニット2では、UNet に追加情報を与えることで、生成される画像の種類をさらにコントロールできることを示しました。これをコンディショニングと呼ぶ。ノイズの多い画像を与えられたモデルは、クラスラベルや安定拡散の場合、画像のテキスト記述などの **追加的な手がかり** に基づいて、ノイズ除去されたバージョンを予測することを課される。推論時には、見たい画像の説明文と、出発点としての純粋なノイズを入力することができ、モデルはランダムな入力をキャプションと一致するものに'ノイズ除去'するために最善を尽くします。 

![text encoder diagram](text_encoder_noborder.png)<br>
_入力されたプロンプトをテキスト埋め込み（encoder_hidden_states）のセットに変換し、UNet に条件付けとして送り込むことができるテキストエンコード処理を示す図です。_

これを実現するためには、テキストを数値で表現し、それが記述する内容に関する関連情報を取得する必要があります。これを実現するために、SD は CLIP と呼ばれるものに基づいて事前に訓練された変換モデルを活用しています。CLIP のテキストエンコーダは、画像のキャプションを画像とテキストの比較に使用できる形式に処理するように設計されているため、画像の説明から有用な表現を作成するタスクに適しています。入力プロンプトは、まずトークン化され（各単語やサブワードに特定のトークンが割り当てられる大規模な語彙に基づいて）、次に CLIP テキストエンコーダを通過して、各トークンに対して768 次元（SD 1.Xの場合）または1024 次元（SD 2.X）ベクトルを生成します。一貫性を保つため、プロンプトは常に77 トークンの長さになるようにパディング/トランケートされるため、条件付けとして使用する最終表現は、プロンプトごとに形状 77 x 1024 のテンソルになります。

![conditioning diagram](sd_unet_color.png)

では、実際にこのコンディショニング情報を UNet に送り込み、UNet が予測を行う際に利用するにはどうすればいいのでしょうか。その答えは、クロスアテンションと呼ばれるものです。UNet にはクロスアテンションレイヤーが散在している。UNet の各空間位置は、テキスト条件付けの異なるトークンに 'attend' して、プロンプトから関連情報を取り込むことができる。上の図は、このテキストコンディショニング（およびタイムステップベースのコンディショニング）が、さまざまなポイントでどのように送り込まれるかを示しています。ご覧のように、どのレベルでも、UNet はこの条件付けを利用する十分な機会があります！

## Classifier-free ガイダンス

It turns out that even with all of the effort put into making the text conditioning as useful as possible, the model still tends to default to relying mostly on the noisy input image rather than the prompt when making its predictions. In a way, this makes sense - many captions are only loosely related to their associated images and so the model learns not to rely too heavily on the descriptions! However, this is undesirable when it comes time to generate new images - if the model doesn't follow the prompt then we may get images out that don't relate to our description at all.

![CFG scale demo grid](cfg_example_0_1_2_10.jpeg)<br>
_Images generated from the prompt "An oil painting of a collie in a top hat" with CFG scale 0, 1, 2 and 10 (left to right)_

To fix this, we use a trick called Classifier-Free Guidance (CGF). During training, text conditioning is sometimes kept blank, forcing the model to learn to denoise images with no text information whatsoever (unconditional generation). Then at inference time, we make two separate predictions: one with the text prompt as conditioning and one without. We can then use the difference between these two predictions to create a final combined prediction that pushes **even further** in the direction indicated by the text-conditioned prediction according to some scaling factor (the guidance scale), hopefully resulting in an image that better matches the prompt. The image above shows the outputs for a prompt at different guidance scales - as you can see, higher values result in images that better match the description.

## その他のコンディショニングの種類: Super-Resolution, Inpainting及びDepth-to-Image

It is possible to create versions of Stable Diffusion that take in additional kinds of conditioning. For example, the [Depth-to-Image model](https://huggingface.co/stabilityai/stable-diffusion-2-depth) has extra input channels that take in-depth information about the image being denoised, and at inference time we can feed in the depth map of a target image (estimated using a separate model) to hopefully generate an image with a similar overall structure.

![depth to image example](https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/depth2image.png)<br>
_Depth-conditioned SD is able to generate different images with the same overall structure (example from StabilityAI)_

In a similar manner, we can feed in a low-resolution image as the conditioning and have the model generate the high-resolution version ([as used by the Stable Diffusion Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)). Finally, we can feed in a mask showing a region of the image to be re-generated as part of the 'in-painting' task, where the non-mask regions need to stay intact while new content is generated for the masked area.

## DreamBooth でのファインチューニング

![dreambooth diagram](https://dreambooth.github.io/DreamBooth_files/teaser_static.jpg)
_Image from the [dreambooth project page](https://dreambooth.github.io/) based on the Imagen model_

DreamBooth is a technique for fine-tuning a text-to-image model to 'teach' it a new concept, such as a specific object or style. The technique was originally developed for Google's Imagen model but was quickly adapted to [work for stable diffusion](https://huggingface.co/docs/diffusers/training/dreambooth). Results can be extremely impressive (if you've seen anyone with an AI profile picture on social media recently the odds are high it came from a dreambooth-based service) but the technique is also sensitive to the settings used, so check out our notebook and [this great investigation into the different training parameters](https://huggingface.co/blog/dreambooth) for some tips on getting it working as well as possible.

## ハンズオンノートブック

| 章                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stable Diffusion Introduction                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit3/01_stable_diffusion_introduction.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit3/01_stable_diffusion_introduction.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit3/01_stable_diffusion_introduction.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit3/01_stable_diffusion_introduction.ipynb)              |
| DreamBooth Hackathon Notebook                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.ipynb)              |
| Stable Diffusion Deep Dive                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb )              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb )              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb )              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb )              |

この時点で、付属のノートブックを使い始めるのに十分な知識があります。上のリンクから好きなプラットフォームで開いてください。 Dreambooth はかなりの計算能力を必要とするので、 Kaggle や Google Colab を使う場合はランタイムタイプを 'GPU' に設定しておくとよいでしょう。

'Stable Diffusion Introduction' ノートブックは、 🤗 Diffusers ライブラリを使った stable diffusion の簡単な紹介で、パイプラインを使って画像を生成・修正する基本的な使用例について説明しています。

DreamBooth Hackathon Notebook （[ハッカソンフォルダ](https://github.com/huggingface/diffusion-models-class/tree/main/hackathon)内）には、新しいスタイルやコンセプトをカバーするモデルのカスタムバージョンを作成するために、自分の画像で SD をファインチューニングする方法が紹介されています。

最後に、 'Stable Diffusion Deep Dive' ノートブックとビデオでは、典型的な生成パイプラインの各ステップを分解し、各ステージを変更してさらに創造的な制御を行うための斬新な方法を提案します。


## プロジェクトタイム

**DreamBooth** ノートブックの指示に従って、指定されたカテゴリの1つについて独自のモデルをトレーニングしてください。各カテゴリで最も優れたモデルを選ぶために、必ず出力例を添付してください。賞品やGPUクレジットなどの詳細については、[ハッカソン情報](https://github.com/huggingface/diffusion-models-class/tree/main/hackathon)をご覧ください。

## 追加資料

- [High-Resolution Image Synthesis with Latent Diffusion Models](http://arxiv.org/abs/2112.10752) - Stable Diffusion の背後にあるアプローチを紹介した論文

- [CLIP](https://openai.com/blog/clip/) - CLIP はテキストと画像の接続を学習し、 CLIP テキストエンコーダーはテキストプロンプトをSDで使用される豊富な数値表現に変換するために使用される。最近のオープンソースの CLIP の亜種（そのうちの1つが SD バージョン2に使われている）の背景については、 [OpenCLIP に関するこの記事](https://wandb.ai/johnowhitaker/openclip-benchmarking/reports/Exploring-OpenCLIP--VmlldzoyOTIzNzIz)も参照してください。

- [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741) テキストコンディショニングと CFG を実証した初期の論文

もっと素晴らしいリソースがありますか？このリストに追加します。
