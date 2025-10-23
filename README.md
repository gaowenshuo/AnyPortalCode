# ANYPORTAL - Official PyTorch Implementation

**ANYPORTAL: Zero-Shot Consistent Video Background Replacement**<br>
[Wenshuo Gao](https://gaowenshuo.github.io/index/), [Xicheng Lan](https://dongfengzy.github.io/) and [Shuai Yang](https://williamyang1991.github.io/)<br>
in ICCV 2025 <br>
[**Project Page**](https://gaowenshuo.github.io/AnyPortal/) | [**Paper**](https://arxiv.org/abs/2509.07472) <br>

**Abstract:** *Despite the rapid advancements in video generation technology, creating high-quality videos that precisely align with user intentions remains a significant challenge. 
Existing methods often fail to achieve fine-grained control over video details, limiting their practical applicability. 
We introduce ANYPORTAL, a novel zero-shot framework for video background replacement that leverages pre-trained diffusion models. 
Our framework collaboratively integrates the temporal prior of video diffusion models with the relighting capabilities of image diffusion models in a zero-shot setting. 
To address the critical challenge of foreground consistency, we propose a Refinement Projection Algorithm, which enables pixel-level detail manipulation to ensure precise foreground preservation. 
ANYPORTAL is training-free and overcomes the challenges of achieving foreground consistency and temporally coherent relighting. 
Experimental results demonstrate that ANYPORTAL achieves high-quality results on consumer-grade GPUs, offering a practical and efficient solution for video content creation and editing.*

<img width="1620" height="658" alt="image" src="https://github.com/user-attachments/assets/84254042-4fbc-4e5f-a1e8-212519cb972b" />

> ⚠️ **Note:**  
> This code is in **development version** for easy debug and modification, which is neither efficient nor user-friendly.  
> We are cleaning our code and a more optimized and convenient version will be released later (before 2026.1.1).  
> 
> ⚠️ **注意：**  
> 本代码目前为适于快速开发和debug的**开发版本**，实现效率较低，使用起来也不够方便。  
> 我们将在今年发布更加优化、易用的版本。

### TODO
- [x] Clean the code for easy use
- [x] Update readme
- [x] ~~Upload paper to arXiv~~
      
## Installation

1. Install [DiffusionAsShader](https://github.com/IGL-HKUST/DiffusionAsShader) under  
   `/src/DiffusionAsShader/` and use its environment.

2. Install the following Python packages:  
   pip install av sympy controlnet-aux omegaconf hydra-core

3. Download weights from [IC-Light](https://github.com/lllyasviel/IC-Light)  
   and place them under `/src/IC-Light/models/`.

4. Install [ProPainter](https://github.com/sczhou/ProPainter) and [MatAnyone](https://github.com/pq-yang/MatAnyone)  
   under `/src/ProPainter/` and `/src/MatAnyone/`, respectively.

## Inference

```bash
sh XXXX.sh
```

## Results

### Text-Guided Background Replacement

https://github.com/user-attachments/assets/1346ea27-87a5-41ee-8eaf-f96d288831b8

### Image-Guided Background Replacement

https://github.com/user-attachments/assets/ad14af81-0fdf-4e61-9603-4e466e528ba4

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{gao2025anyportal,
 title = {ANYPORTAL: Zero-Shot Consistent Video Background Replacement},
 author = {Gao, Wenshuo and Lan, Xicheng and Yang, Shuai},
 booktitle = {ICCV},
 year = {2025},
}
```

## Acknowledgments

The code is mainly developed based on [IC-Light](https://github.com/lllyasviel/IC-Light), [CogVideoX](https://github.com/zai-org/CogVideo), [DiffusionAsShader](https://github.com/IGL-HKUST/DiffusionAsShader), [ProPainter](https://github.com/sczhou/ProPainter) and [MatAnyone](https://github.com/pq-yang/MatAnyone).
