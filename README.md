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

> ⚠️ **Note:**  
> We apologize that this code is in **development version** for easy debug and modification, which is neither efficient nor user-friendly.  
> We are cleaning our code and a more optimized and convenient version will be released later (before 2026.1.1).  
> 
> ⚠️ **注意：**  
> 我们真诚的抱歉，本代码目前为适于快速开发和debug的**开发版本**，实现效率较低，使用起来也不够方便。  
> 我们将在今年发布更加优化、易用的版本。  


## Installation

1. Install [DiffusionAsShader](https://github.com/IGL-HKUST/DiffusionAsShader) under  
   `/src/DiffusionAsShader/` and use its environment.

2. Install the following Python packages:  
   pip install av sympy controlnet-aux omegaconf hydra-core

3. Download weights from [IC-Light](https://github.com/lllyasviel/IC-Light)  
   and place them under `/src/IC-Light/models/`.

4. Install [ProPainter](https://github.com/sczhou/ProPainter) and [MatAnyone](https://github.com/pq-yang/MatAnyone)  
   under `/src/ProPainter/` and `/src/MatAnyone/`, respectively.
