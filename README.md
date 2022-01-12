# Texture Reformer
**[update 1/12/2022]**

Official Pytorch code for ["Texture Reformer: Towards Fast and Universal Interactive Texture Transfer"](https://arxiv.org/abs/2112.02788) (AAAI 2022)

## Introduction:

**Texture reformer is the first fast and universal neural-based framework for interactive texture transfer with user-specified guidance.** It uses a feed-forward multi-view and multi-stage synthesis procedure consisting of three different stages: I) a global view structure alignment stage, II) a local view texture refinement stage, and III) a holistic effect enhancement stage. Moreover, a novel learning-free view-specific texture reformation (VSTR) operation with a new semantic map guidance strategy is also presented to realize more accurate semantic-guided and structure-preserved texture transfer. 

You can apply texture reformer to any **interactive texture transfer** application scenarios, including doodles-to-artworks, texture pattern editing, text effects transfer, and virtual clothing manipulation. Moreover, we also extend it to [**semantic style transfer**](https://arxiv.org/pdf/1603.01768.pdf), which allows the users to provide an additional content image as input.

![show](https://github.com/EndyWon/Texture-Reformer/blob/main/figures/teaser.jpg)

## Environment:
- Python 3.6
- Pytorch 1.4.0 (strongly recommended!!!)
- Other needed libraries are summarized in `requirements.txt`. Simply install them by `pip install -r requirements`

## Getting Started:
**Step 1: Clone this repo**

`git clone https://github.com/EndyWon/Texture-Reformer`  
`cd Texture-Reformer`

**Step 2: Prepare models**

- Download the pre-trained auto-encoder models from this [google drive](https://drive.google.com/file/d/13n_YJ6J8lIvF-liWFeJY35nXsZM-5vTZ/view?usp=sharing). Unzip and place them at path `models/`.
- We also provide the **small** pre-trained models compressed by [Collaborative-Distillation](https://github.com/MingSun-Tse/Collaborative-Distillation) in this [google drive](https://drive.google.com/file/d/1RkDJs6Hv7FQ-vdw9B9qDzrzq8l79dABS/view?usp=sharing). Unzip and place them at path `small_models/`.

**Step 3: Run transfer script**

- For **interactive texture transfer**, you only need to input **three** images: the source image `-style`, the semantic map of source image `-style_sem`, and the semantic map of target image `-content_sem`, like follows:

`python transfer.py -content_sem inputs/doodles/Seth_sem.png -style inputs/doodles/Gogh.jpg -style_sem inputs/doodles/Gogh_sem.png`

![show](https://github.com/EndyWon/Texture-Reformer/blob/main/figures/texture_transfer.jpg)

- For **semantic style transfer**, you need to first activate the style transfer mode by `-style_transfer`, and then input **four** images: the content image `-content`, the style image `-style`, the semantic map of content image `-content_sem` (not necessary), and the semantic map of style image `-style_sem` (not necessary), like follows:

`python transfer.py -style_transfer -content inputs/doodles/Seth.jpg -content_sem inputs/doodles/Seth_sem.png -style inputs/doodles/Gogh.jpg -style_sem inputs/doodles/Gogh_sem.png -coarse_alpha 0.5 -fine_alpha 0.5`

![show](https://github.com/EndyWon/Texture-Reformer/blob/main/figures/style_transfer.jpg)


## Script Parameters:

**Specify inputs and outputs**

- `-content` : File path to the content image, valid for style transfer and invalid for texture transfer.
- `-style` : File path to the style/source image.
- `-content_sem` : File path to the semantic map of content/target image.
- `-style_sem` : File path to the semantic map of style/source image.
- `-outf` : Folder to save output images.
- `-content_size` : Resize content/target.
- `-style_size` : Resize style/source.
- `-style_transfer` : Activate it if you want style transfer rather than texture transfer.

**Runtime controls**

- `-coarse_alpha` : Hyperparameter to blend transformed feature with content feature in coarse level (level 5).
- `-fine_alpha` : Hyperparameter to blend transformed feature with content feature in fine level (level 4).
- `-semantic` : Choose different modes to embed semantic maps, choices = (`add`, `concat`, `concat_ds`). `add`: addition, `concat`: concatenation, `concat_ds`: concat downsampled semantic maps.
- `-concat_weight` : Hyperparameter to control the semantic guidance/awareness weight for `-semantic concat` mode and `-semantic concat_ds` mode, range `0-inf`.
- `-add_weight` : Hyperparameter to control the semantic guidance/awareness weight for `-semantic add` mode, range `0-1`.
- `-coarse_psize` : Patch size in coarse level (level 5), `0` means using global view.
- `-fine_psize` : Patch size in fine level (level 4).
- `-enhance` : Choose different enhancement modes in level 3, level 2, and level 1, choices = (`adain`, `wct`). `adain`: first-order statistics enhancement, `wct`: second-order statistics enhancement.
- `-enhance_alpha` : Hyperparameter to control the enhancement degree in level 3, level 2, and level 1.

**Compress Models**
- `-compress` : Use the compressed models for faster inference.
