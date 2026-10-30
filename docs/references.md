# References

The licenses listed were checked in October 2026. Re-check them before
relying on any of them.

## Architecture
- NAFNet: Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022. https://github.com/megvii-research/NAFNet (MIT). SIDD-w32 config: `options/train/SIDD/NAFNet-width32.yml`
- Restormer (MIT): https://github.com/swz30/Restormer

## Noise modelling
- ELD: Wei et al., CVPR 2020 / TPAMI 2021. https://arxiv.org/abs/2003.12751 · https://github.com/Vandermode/ELD
- SFRN: Zhang et al., ICCV 2021. https://arxiv.org/abs/2110.04756
- PMN + LRID: Feng et al., ACM MM 2022 / TPAMI. https://arxiv.org/abs/2207.06103 · https://huggingface.co/datasets/hansen97/LRID (CC BY-NC)
- PNNP: https://arxiv.org/abs/2310.09126 · https://github.com/fenghansen/PNNP (Apache-2.0)
- LRD, general noise synthesis: Zhang et al., ICCV 2023. https://github.com/fengzhang427/lrd
- LED, Lighting Every Darkness: https://github.com/Srameo/LED (CC BY-NC)
- Noise Flow: Abdelhamed et al., ICCV 2019
- Noise Modeling in One Hour: Li, Jiang, Iso (Sony), CVPR 2025. https://arxiv.org/abs/2505.00045 · https://github.com/SonyResearch/raw_image_denoising (MIT)
- Dark Noise Diffusion (2025 preprint): https://arxiv.org/abs/2503.11262
- SNIC (2025–26): https://arxiv.org/abs/2512.15905
- AIM 2025 Real-World RAW Denoising Challenge: https://arxiv.org/abs/2510.06601 · https://sonyresearch.github.io/AIM2025_Denoise_Challenge/
- NTIRE 2025 RAW restoration: https://arxiv.org/abs/2506.02197
- Unprocessing: Brooks et al., CVPR 2019. https://arxiv.org/abs/1811.11127

## Self-supervised denoising
- Neighbor2Neighbor: https://arxiv.org/abs/2101.02824
- AP-BSN: https://arxiv.org/abs/2203.11799

## Bayer augmentation
- BayerUnify / BayerAug: Liu et al., CVPRW 2019. https://arxiv.org/abs/1904.12945

## Secondary track: RAW super-resolution
- Wronski et al., *Handheld Multi-Frame Super-Resolution*, SIGGRAPH 2019. Joint reconstruction of linear RGB from shifted CFA RAW frames. https://sites.google.com/view/handheld-super-res/
- Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016. PixelShuffle / learned upsampling; not additional optical measurements. https://openaccess.thecvf.com/content_cvpr_2016/html/Shi_Real-Time_Single_Image_CVPR_2016_paper.html
- Xu et al., *Towards Real Scene Super-Resolution with Raw Images*, CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Towards_Real_Scene_Super-Resolution_With_Raw_Images_CVPR_2019_paper.pdf

## Datasets
- SIDD (MIT): https://abdokamel.github.io/sidd/
- SID: https://github.com/cchen156/Learning-to-See-in-the-Dark
- RawNIND (CC BY-SA 4.0): https://arxiv.org/abs/2501.08924 · doi:10.14428/DVN/DEQCIM
- raw.pixls.us (CC0): https://raw.pixls.us/
- MCR: https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark
- RAISE: https://loki.disi.unitn.it/RAISE/
- MIT-Adobe FiveK: https://data.csail.mit.edu/graphics/fivek/
- DND: https://noise.visinf.tu-darmstadt.de/

## Calibration
- EMVA 1288, the photon transfer method: https://image-engineering.de/library/technotes/738-emva-1288

## Core ML / ANE
- Typed execution (fp16): https://apple.github.io/coremltools/docs-guides/source/typed-execution.html
- ANE deployment guidance: https://machinelearning.apple.com/research/neural-engine-transformers
- LayerNorm fp16 on the ANE: https://developer.apple.com/forums/thread/728931
- Dynamic shapes and the ANE: https://developer.apple.com/forums/thread/724930
