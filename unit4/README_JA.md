# ユニット4: Diffusion をもっと使いこなす

Hugging Face Diffusion モデルコースのユニット4へようこそ! このユニットでは、最新の研究で登場した diffusion モデルの多くの改良と拡張のいくつかを見ていきます。これまでのユニットよりもコード量が少なく、さらなる研究のための出発点となるように設計されています。

## このユニットを開始する :rocket:

ユニットの手順は以下の通りです:

- [このコースに申し込んだ](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)ことを確認し、コースにユニットが追加されたときに通知されるようにします
- このユニットで扱われるさまざまなトピックの概要については、以下の資料に目を通してください
- リンク先の動画や資料で、特定のトピックをより深く知ることができます
- デモノートを見て、 'What Next' セクションを読んで、プロジェクトの提案をしてください

:loudspeaker: [Discord](https://huggingface.co/join/discord) への参加もお忘れなく。ここでは、教材について議論したり、作ったものを `#diffusion-models-class` チャンネルで共有することができます。

## 目次

- [Unit 4: Going Further with Diffusion Models](#unit-4-going-further-with-diffusion-models)
  - [Start this Unit :rocket:](#start-this-unit-rocket)
  - [Table of Contents](#table-of-contents)
  - [Faster Sampling via Distillation](#faster-sampling-via-distillation)
  - [Training Improvements](#training-improvements)
  - [More Control for Generation and Editing](#more-control-for-generation-and-editing)
  - [Video](#video)
  - [Audio](#audio)
  - [New Architectures and Approaches - Towards 'Iterative Refinement'](#new-architectures-and-approaches---towards-iterative-refinement)
  - [Hands-On Notebooks](#hands-on-notebooks)
  - [Where Next?](#where-next)


## ディスティレーションによるサンプリングの高速化

Progressive Distillation とは、既存の diffusion モデルを用いて、より少ないステップで推論を行う新しいバージョンのモデルを学習させる手法である。'student' モデルは 'teacher' モデルの重みをもとに初期化される。学習中、教師モデルは2回のサンプリングステップを行い、生徒モデルは1回のステップで結果の予測に一致させようとする。このプロセスは複数回繰り返すことができ、前の反復の学生モデルが次のステージの教師となる。その結果、元の教師モデルよりもはるかに少ないステップ（通常4または8）で適切なサンプルを作成することができるモデルができました。コアとなるメカニズムは、[このアイデアを紹介した論文](http://arxiv.org/abs/2202.00512)に掲載されたこの図に示されています:

![image](https://user-images.githubusercontent.com/6575163/211016659-7dac24a5-37e2-45f9-aba8-0c573937e7fb.png)

_Progressive Distillation illustrated (from the [paper](http://arxiv.org/abs/2202.00512))_

既存のモデルを使って新しいモデルを「教える」というアイデアは、教師モデルによって分類器を使わないガイダンス技術が使用され、生徒モデルは目標とするガイダンススケールを指定する追加入力に基づいて、1つのステップで同等の出力を生成するように学習しなければならない、ガイド付きモデルを作るために拡張することができます。これにより、高品質なサンプルを作成するために必要なモデル評価回数をさらに減らすことができます。[このビデオ](https://www.youtube.com/watch?v=ZXuK6IRJlnk)では、この手法の概要を紹介しています。

NB: A distilled version of Stable Diffusion is due to be released fairly soon.

Key references:
- [Progressive Distillation For Fast Sampling Of Diffusion Models](http://arxiv.org/abs/2202.00512)
- [On Distillation Of Guided Diffusion Models](http://arxiv.org/abs/2210.03142)

## トレーニングの改善

There have been several additional tricks developed to improve diffusion model training. In this section we've tried to capture the core ideas from recent papers. There is a constant stream of research coming out with additional improvements, so if you see a paper you feel should be added here please let us know!

![image](https://user-images.githubusercontent.com/6575163/211021220-e87ca296-cf15-4262-9359-7aeffeecbaae.png)
_Figure 2 from the [ERNIE-ViLG 2.0 paper](http://arxiv.org/abs/2210.15257)_

Key training improvements:
- Tuning the noise schedule, loss weighting and sampling trajectories for more efficient training. An excellent paper exploring some of these design choices is [Elucidating the Design Space of Diffusion-Based Generative Models](http://arxiv.org/abs/2206.00364) by Karras et al.
- Training on diverse aspect ratios, as described in [this video from the course launch event](https://www.youtube.com/watch?v=g6tIUrMvOec).
- Cascaded diffusion models, training one model at low resolution and then one or more super-res models. Used in DALLE-2, Imagen and more for high-resolution image generation.
- Better conditioning, incorporating rich text embeddings ([Imagen](https://arxiv.org/abs/2205.11487) uses a large language model called T5) or multiple types of conditioning ([eDiffi](http://arxiv.org/abs/2211.01324))
- 'Knowledge Enhancement' - incorporating pre-trained image captioning and object detection models into the training process to create more informative captions and produce better performance ([ERNIE-ViLG 2.0](http://arxiv.org/abs/2210.15257))
- 'Mixture of Denoising Experts' (MoDE) - training different variants of the model ('experts') for different noise levels as illustrated in the image above from the [ERNIE-ViLG 2.0 paper](http://arxiv.org/abs/2210.15257).

Key references:
- [Elucidating the Design Space of Diffusion-Based Generative Models](http://arxiv.org/abs/2206.00364)
- [eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](http://arxiv.org/abs/2211.01324)
- [ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts](http://arxiv.org/abs/2210.15257)
- [Imagen - Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) ([demo site](https://imagen.research.google/))

## More Control for Generation and Editing

In addition to training improvements, there have been several innovations in the sampling and inference phase, including many approaches that can add new capabilities to existing diffusion models.

![image](https://user-images.githubusercontent.com/6575163/212529129-3de41cf4-6f70-4607-8448-e9bbe9d190cf.png)
_Samples generated by 'paint-with-words' ([eDiffi](http://arxiv.org/abs/2211.01324))_

The video ['Editing Images with Diffusion Models'](https://www.youtube.com/watch?v=zcG7tG3xS3s) gives an overview of the different methods being used to edit existing images with diffusion models. The available techniques can be split into four main categories:

1) Add noise and then denoise with a new prompt. This is the idea behind the `img2img` pipeline, which has been modified and extended in various papers:
- [SDEdit](https://sde-image-editing.github.io/) and [MagicMix](https://magicmix.github.io/) build on this idea
- DDIM inversion (TODO link tutorial) uses the model to 'reverse' the sampling trajectory rather than adding random noise, giving more control
- [Null-text Inversion](https://null-text-inversion.github.io/) enhances the performance of this kind of approach dramatically by optimizing the unconditional text embeddings used for classifier-free guidance at each step, allowing for extremely high-quality text-based image editing.
2) Extending the ideas in (1) but with a mask to control where the effect is applied
- [Blended Diffusion](https://omriavrahami.com/blended-diffusion-page/) introduces the basic idea
- [This demo](https://huggingface.co/spaces/nielsr/text-based-inpainting) uses an existing segmentation model (CLIPSeg) to create the mask based on a text description
- [DiffEdit](https://arxiv.org/abs/2210.11427) is an excellent paper that shows how the diffusion model itself can be used to generate an appropriate mask for editing the image based on text.
- [SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model](https://arxiv.org/abs/2212.05034) fine-tunes a diffusion model for more accurate mask-guided inpainting.
3) Cross-attention Control: using the cross-attention mechanism in diffusion models to control the spatial location of edits for more fine-grained control.
- [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626) is the key paper that introduced this idea, and the technique has [since been applied to Stable Diffusion](https://wandb.ai/wandb/cross-attention-control/reports/Improving-Generative-Images-with-Instructions-Prompt-to-Prompt-Image-Editing-with-Cross-Attention-Control--VmlldzoyNjk2MDAy)
- This idea is also used for 'paint-with-words' ([eDiffi](http://arxiv.org/abs/2211.01324), shown above)
4) Fine-tune ('overfit') on a single image and then generate with the fine-tuned model. The following papers both published variants of this idea at roughly the same time:
- [Imagic: Text-Based Real Image Editing with Diffusion Models](https://arxiv.org/abs/2210.09276)
- [UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image
](https://arxiv.org/abs/2210.09477)

The paper [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800) is notable in that it used some of the image editing techniques described above to build a synthetic dataset of image pairs alongside image edit instructions (generated with GPT3.5) to train a new model capable of editing images based on natural language instructions


## ビデオ

![image](https://user-images.githubusercontent.com/6575163/213657523-be40178a-4357-410b-89e3-a4cbd8528900.png)
_Still frames from [sample videos generated with Imagen Video](https://imagen.research.google/video/)_

A video can be represented as a sequence of images, and the core ideas of diffusion models can be applied to these sequences. Recent work has focused on finding appropriate architectures (such as '3D UNets' which operate on entire sequences) and on working efficiently with video data. Since high-frame-rate video involves a lot more data than still images, current approaches tend to first generate low-resolution and low-frame-rate video and then apply spatial and temporal super-resolution to produce the final high-quality video outputs.

Key references:
- [Video Diffusion Models](https://video-diffusion.github.io/)
- [IMAGEN VIDEO: HIGH DEFINITION VIDEO GENERATION WITH DIFFUSION MODELS](https://imagen.research.google/video/paper.pdf)

## Audio

![image](https://user-images.githubusercontent.com/6575163/213657272-a1b54017-216f-453b-9b28-97c6fef21f54.png)

_A spectrogram generated with Riffusion ([image source](https://www.riffusion.com/about))_

While there has been some work on generating audio directly using diffusion models (e.g. [DiffWave](https://arxiv.org/abs/2009.09761)) the most successful approach so far has been to convert the audio signal into something called a spectrogram, which effectively 'encodes' the audio as a 2D "image" which can then be used to train the kinds of diffusion models we're used to using for image generation. The resulting generated spectrograms can then be converted into audio using existing methods. This approach is behind the recently-released Riffusion, which fine-tuned Stable Diffusion to generate spectrograms conditioned on text - [try it out here](https://www.riffusion.com/).

The field of audio generation is moving extremely quickly. Over the past week (at the time of writing) there have been at least 5 new advances announced, which are marked with a star in the list below:

Key references:
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761)
- ['Riffusion'](https://www.riffusion.com/about) (and [code](https://github.com/riffusion/riffusion))
- *[MusicLM](https://google-research.github.io/seanet/musiclm/examples/) by Google generates consistent audio from text and can be conditioned with hummed or whistled melodies
- *[RAVE2](https://github.com/acids-ircam/RAVE) - a new version of a Variational Auto-Encoder that will be useful for latent diffusion on audio tasks. This is used in the soon-to-be-announced *[AudioLDM](https://twitter.com/LiuHaohe/status/1619119637660327936?s=20&t=jMkPWBFuAH19HI9m5Sklmg) model
- *[Noise2Music](https://noise2music.github.io/) - A diffusion model trained to produce high-quality 30-second clips of audio based on text descriptions
- *[Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models](https://text-to-audio.github.io/) - a diffusion model trained to generate diverse sounds based on text
- *[Moûsai: Text-to-Music Generation with Long-Context Latent Diffusion](https://arxiv.org/abs/2301.11757)

## New Architectures and Approaches - Towards 'Iterative Refinement'

![image](https://user-images.githubusercontent.com/6575163/213731066-0fbe38a7-233f-42be-99fc-38cea889c86b.png)

_Figure 1 from the [Cold Diffusion](http://arxiv.org/abs/2208.09392) paper_

We are slowly moving beyond the original narrow definition of "diffusion" models and towards a more general class of models that perform **iterative refinement**, where some form of corruption (like the addition of gaussian noise in the forward diffusion process) is gradually reversed to generate samples. The 'Cold Diffusion' paper demonstrated that many other types of corruption can be iteratively 'undone' to generate images (examples shown above), and recent transformer-based approaches have demonstrated the effectiveness of token replacement or masking as a noising strategy.

![image](https://user-images.githubusercontent.com/6575163/213731351-7fd6c98c-6ba6-4bd9-a898-230002fc334f.png)

_Pipeline from [MaskGIT](http://arxiv.org/abs/2202.04200)_

The UNet architecture at the heart of many current diffusion models is also being replaced with different alternatives, most notably various transformer-based architectures. In [Scalable Diffusion Models with Transformers (DiT)](https://www.wpeebles.com/DiT) a transformer is used in place of the UNet for a fairly standard diffusion model approach, with excellent results. [Recurrent Interface Networks](https://arxiv.org/pdf/2212.11972.pdf) applies a novel transformer-based architecture and training strategy in pursuit of additional efficiency. [MaskGIT](http://arxiv.org/abs/2202.04200) and [MUSE](http://arxiv.org/abs/2301.00704) use transformer models to work with tokenized representations of images, although the [Paella](https://arxiv.org/abs/2211.07292v1) model demonstrates that a UNet can also be applied successfully to these token-based regimes too.

With each new paper, more efficient approaches are being developed, and it may be some time before we see what peak performance looks like on these kinds of iterative refinement tasks. There is much more still to explore!

Key references

- [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](http://arxiv.org/abs/2208.09392)
- [Scalable Diffusion Models with Transformers (DiT)](https://www.wpeebles.com/DiT)
- [MaskGIT: Masked Generative Image Transformer](http://arxiv.org/abs/2202.04200)
- [Muse: Text-To-Image Generation via Masked Generative Transformers](http://arxiv.org/abs/2301.00704)
- [Fast Text-Conditional Discrete Denoising on Vector-Quantized Latent Spaces (Paella)](https://arxiv.org/abs/2211.07292v1)
- [Recurrent Interface Networks](https://arxiv.org/pdf/2212.11972.pdf) - a promising new architecture that does well at generating high-resolution images without relying on latent diffusion or super-resolution. See also, [simple diffusion: End-to-end diffusion for high-resolution images](https://arxiv.org/abs/2301.11093) which highlights the importance of the noise schedule for training at higher resolutions.

## ハンズオンノートブック

| 章                                     | Colab                                                                                                                                                                                               | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DDIM Inversion                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)              |
| Diffusion for Audio                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit4/02_diffusion_for_audio.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit4/02_diffusion_for_audio.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit4/02_diffusion_for_audio.ipynb)              | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit4/02_diffusion_for_audio.ipynb)              |

We've covered a LOT of different ideas in this unit, many of which deserve much more detailed follow-on lessons in the future. For now, you can two of the many topics via the hands-on notebooks we've prepared.
- **DDIM Inversion** shows how a technique called inversion can be used to edit images using existing diffusion models
- **Diffusion for Audio** introduces the idea of spectrograms and shows a minimal example of fine-tuning an audio diffusion model on a specific genre of music.

## 次はどこ？

これがこのコースのとりあえずの最終ユニットであり、つまり、次に何が来るかはあなた次第なのです！ Hugging Face [Discord](https://huggingface.co/join/discord) では、いつでも質問やプロジェクトについてのチャットができることを忘れないでください。どんな作品が出来上がるか楽しみです 🤗
