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

- Download the pre-trained auto-encoder models from this [google drive]().


