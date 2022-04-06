---
layout:     post
author:     "Zonghan Yang"
title:      "On Robust Prefix-Tuning for Text Classification"
date:       2022-04-05 12:00:00
summary:    Prefix-tuning lacks robustness, while current defense methods will hamper the modularity of the prefixes. We tune an additional prefix during inference to steer correct activation of the pretrained LM, which significantly improves the robustness.
image:      "/images/robust-prefix-tuning/figure-1-static.png"
---

Prefix-tuning ([Li & Liang, 2021](https://arxiv.org/abs/2101.00190)) is going viral. Instead of finetuning the entire large-scale language model (LM), all you need is a prefix when adapting to a certain downstream task. Should I apologize for abusing the "all you need is XXX" pattern, but there's indeed many merits in it: more **expressive** than discrete prompts (which is enabled by continuous optimization), **lightweight** compared with finetuning (near 1000x few parameters to be updated), as well as **modular** (one prefix one task). Without a doubt, prefix-tuning will motivate us to consider more about adapting tasks to the pretrained LMs in this era of parameter-efficient finetuning ([Liu et al., 2021](https://arxiv.org/abs/2107.13586); [Ding et al., 2022](https://arxiv.org/abs/2203.06904)).

We played with prefix-tuning on text classification tasks. Despite all the merits in prefix-tuning, we found it lacking in robustness. It's easy to fool the prefix by manipulating the inputs (see an example below). In hindsight, perhaps we shouldn't be surprised by such phenomenon ([Ilyas et al., 2019](https://arxiv.org/abs/1905.02175)) - Defending against adversarial attacks for prefix-tuning is what we are gonna do now, as we want parameter-efficient finetuning techniques to be robust.

<br>
<div align="center"><img src="{{ site.url }}/images/robust-prefix-tuning/attack-example.png" width=700></div>
<br>

But we got really surprised when looking at the current defense methods in NLP. Roughly speaking, four types of defense methods have been proposed: model functional improvement ([Li & Sethy, 2019](); [Jone et al.,2020]()), robustness certification ([Jia et al., 2019](); [Shi et al., 2020]()), adversary detection ([Pruthi et al., 2019](); [Zhou et al., 2019]()), as well as adversarial training ([Miyato et al., 2017](); [Miyato et al., 2019](); [Zhu et al., 2020](); [Yi et al., 2021]()). There are many more papers not listed here, but most of the techniques require *modification to the architecture and the parameters of the LM* or additional maintenance of adversary detectors. 

Directly applying such techniques necessitates auxiliary model update and storage, which will inevitably hamper the modularity of prefix-tuning. It seems that adversarial training is the only exception, but it is notorious for its excessively long training time (and we'll soon find it not suitable in the prefix-tuning scenario). We ask the following question:

*Can we improve the robustness of prefix-tuning while preserving its efficiency and modularity, without modifying the pretrained model parameters?*

Our answer is **YES**. Let me show you what we do in our ICLR 2022 [paper](https://openreview.net/forum?id=eBCmOocUejf).

## <span id="design-robust-prefixes">Designing Robust Prefixes</span>

How do we achieve both robustness and parameter efficiency?

<div align="center"><img src="{{ site.url }}/images/robust-prefix-tuning/observation.png" width=700></div>

To answer this question, we first observe how prefix-tuning steers a GPT-style LM to classify a sentence. Causal LMs cast the text classification problem as label generation. When classifying, the LM generates the predicted label token at the output position (with a special token \[ANS\] as input) after autoregressively processing all tokens in the sentence. Now focus on the output position: while the input token is the same (\[ANS\]), the output prediction token differs according to different contexts. To be more specific, it's the affected layerwise activation of the LM that results in different model bahaviour at the output position. As a result, we might want to rectify the erroneous activation in case the model is fooled.

So how do we rectify the activation? Recall that prefix-tuning steers the activation of LMs. In the same vein, we can expect a prefix to rectify the erroneous activation when optimized properly. In our work, we propose to tune an another prefix $$P'_\Psi$$ to achieve this during inference, with the original prefix $$P_\theta$$ fixed. We expect the added prefix $$P_\theta + P'_\Psi$$ will steer correct activations under attacks. Now all we need is the additional prefix for robustness: no need to modify the pretrained LM, which preserves the parameter efficiency. To tune the additional prefix $$P'_\Psi$$, we assume that all the correct activations at layer $$j$$ lie on a manifold $$\mathcal{M}^{(j)}$$, and propose to minimize the orthogonal component of an erroneous activation. Our method is illustrated as follows:

<br>
<div align="center"><img src="{{ site.url }}/images/robust-prefix-tuning/figure-1-static.png" width=700></div>
<br>

We construct the canonical manifold by PCA to characterize the layerwise activation of the correctly-classified inputs. For the $$j$$-th layer, we obtain a layerwise linear projection matrix $$Q^{(j)}$$. When prompted with $$P_\theta + P'_\Psi$$ (with $$P'_\Psi$$ initialized as $$\mathbf{0}$$), the $$j$$-th layer activation at the output position $$o$$ is $$h_{o}^{(j)}$$, and

$$ \| h_{o}^{(j)} - h_{o}^{(j)} Q^{(j)} \|_2 $$

defines the loss function to update $$P'_\Psi$$ on the fly.

## Performance of the Robust Prefix

We implement both standard and adversarial training to obtain the original prefix $$P_\theta$$ and experiment on three datasets: SST-2, AG's News, and SNLI. With properly tuned robust prefix $$P'_\Psi$$, the robustness of both standard and adversarial prefix-tuning is significantly improved. Check the results from SST-2 for example:

<div align="center"><img src="{{ site.url }}/images/robust-prefix-tuning/main-results.png" width=700></div>

It is also noted that standard prefix-tuning, when equipped with the robust prefix, outperforms adversarial prefix-tuning without robust prefix in terms of robustness (right bar in the left figure v.s. orange bar in the right figure). This suggests that our robust prefix-tuning might be a better solution compared with adversarial training. We are further convinced of this when comparing standard and adversarial prefix-tuning:

<div align="center"><img src="{{ site.url }}/images/robust-prefix-tuning/epoch-time-comparison.png" width=700></div>

As you can find in the left figure, adversarial prefix-tuning achieves slightly better **epoch**-wise convergence and generalization. With the x-axis replaced with **clocktime**, however, it is shown that adversarial prefix-tuning requires much longer time to converge.

## Behavior Analysis

The summed prefix $$P_\theta + P'_\Psi$$ steers the correct activation of the LM, which dramatically improves the robustness of prefix-tuning. Compared with the original prefix $$P_\theta$$, how does the robust prefix steer the LM?

TBA soon...


## Takeaway Message

**We tune an additional prefix** (which preserves the strengths of the original prefix-tuning method) **during inference** (which avoids the computational cost of adversarial training) **to steer correct activation** (refining the representation at the output position in the pretrained LM). 

Our code is available at [https://github.com/minicheshire/Robust-Prefix-Tuning](https://github.com/minicheshire/Robust-Prefix-Tuning). Future work includes but is not limited to constructing a better canonical manifold, generalizing the method to text generation, etc.

## Contact
For further discussion, please contact: **[Zonghan Yang](https://minicheshire.github.io/)** (*yangzh20@mails.tsinghua.edu.cn*).