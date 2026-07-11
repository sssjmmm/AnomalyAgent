# Third-party code

`metauas.py` is derived from the official [MetaUAS repository](https://github.com/gaobb/MetaUAS), copyright Bin-Bin Gao, and is distributed under the MIT license in `LICENSE.MetaUAS`.

The downstream classification/localization utilities in `evaluation/` are adapted from the official [AnomalyDiffusion repository](https://github.com/sjtuplayer/anomalydiffusion), distributed under the MIT license reproduced in `LICENSE.AnomalyDiffusion`.

The MetaUAS checkpoint is not included. Download it from the official project and set `METAUAS_CKPT`. LLaMA-Factory, OpenRLHF, Qwen3-VL, Gemini services, MVTec AD, and other dependencies are also not vendored; their own licenses and terms apply.
