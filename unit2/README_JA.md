# ユニット2：ファインチューニング、ガイダンス、コンディショニング

Hugging Face Diffusion モデルコースのユニット2へようこそ! このユニットでは、事前にトレーニングされた diffusion モデルを新しい方法で使用し、適応させる方法を学びます。また、生成プロセスを制御するために、**コンディショニング** として追加の入力を受ける diffusion モデルをどのように作成するかもご覧いただきます。

## このユニットを開始する :rocket:

ユニットの手順は以下の通りです:

- Make sure you've [signed up for this course](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) so that you can be notified when new material is released
- Read through the material below for an overview of the key ideas of this unit
- Check out the _**Fine-tuning and Guidance**_ notebook to fine-tune an existing diffusion model on a new dataset using the 🤗 Diffusers library and to modify the sampling procedure using guidance
- Follow the example in the notebook to share a Gradio demo for your custom model
- (Optional) Check out the _**Class-conditioned Diffusion Model Example**_ notebook to see how we can add additional control to the generation process.
- (Optional) Check out [this video](https://www.youtube.com/watch?v=mY20iKOQ2zw) for an informal run-through of the material in this unit.


:loudspeaker: Don't forget to join the [Discord](https://huggingface.co/join/discord), where you can discuss the material and share what you've made in the `#diffusion-models-class` channel.

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

There are a number of ways to pass in this conditioning information, such as
- Feeding it in as additional channels in the input to the UNet. This is often used when the conditioning information is the same shape as the image, such as a segmentation mask, a depth map or a blurry version of the image (in the case of a restoration/superresolution model). It does work for other types of conditioning too. For example, in the notebook, the class label is mapped to an embedding and then expanded to be the same width and height as the input image so that it can be fed in as additional channels.
- Creating an embedding and then projecting it down to a size that matches the number of channels at the output of one or more internal layers of the UNet, and then adding it to those outputs. This is how the timestep conditioning is handled, for example. The output of each Resnet block has a projected timestep embedding added to it. This is useful when you have a vector such as a CLIP image embedding as your conditioning information. A notable example is the ['Image Variations' version of Stable Diffusion](https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations) which does exactly this.
- Adding cross-attention layers that can 'attend' to a sequence passed in as conditioning. This is most useful when the conditioning is in the form of some text - the text is mapped to a sequence of embeddings using a transformer model, and then cross-attention layers in the UNet are used to incorporate this information into the denoising path. We'll see this in action in Unit 3 as we examine how Stable Diffusion handles text conditioning.


## ハンズオンノートブック

| Chapter                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fine-tuning and Guidance                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              |
| Class-conditioned Diffusion Model Example                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              |

At this point, you know enough to get started with the accompanying notebooks! Open them in your platform of choice using the links above. Fine-tuning is quite computationally intensive, so if you're using Kaggle or Google Colab make sure you set the runtime type to 'GPU' for the best results.

The bulk of the material is in _**Fine-tuning and Guidance**_, where we explore these two topics through worked examples. The notebook shows how you can fine-tune an existing model on new data, add guidance, and share the result as a Gradio demo. There is an accompanying script ([finetune_model.py](https://github.com/huggingface/diffusion-models-class/blob/main/unit2/finetune_model.py)) that makes it easy to experiment with different fine-tuning settings, and [an [example space](https://huggingface.co/spaces/johnowhitaker/color-guided-wikiart-diffusion) that you can use as a template for sharing your own demo on 🤗 Spaces.

In the _**Class-conditioned Diffusion Model Example**_, we show a brief worked example of creating a diffusion model conditioned on class labels using the MNIST dataset. The focus is on demonstrating the core idea as simply as possible: by giving the model extra information about what it is supposed to be denoising, we can later control what kinds of images are generated at inference time.

## プロジェクトタイム

Following the examples in the _**Fine-tuning and Guidance**_ notebook, fine-tune your own model or pick an existing model and create a Gradio demo to showcase your new guidance skills. Don't forget to share your demo on Discord, Twitter etc so we can admire your work!

## 追加資料

[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) - Introduced the DDIM sampling method (used by DDIMScheduler)

[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741) - Introduced methods for conditioning diffusion models on text

[eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://arxiv.org/abs/2211.01324) - Shows how many different kinds of conditioning can be used together to give even more control over the kinds of samples generated

もっと素晴らしいリソースがありますか？このリストに追加します。
