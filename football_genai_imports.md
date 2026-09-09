# Football GenAI — Python Libraries

## 1. Core Deep Learning

### PyTorch
```bash
pip install torch torchvision torchaudio
```

```python
import torch
import torchvision
```

### Diffusers
```bash
pip install diffusers
```

```python
import diffusers
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
```

### Transformers
```bash
pip install transformers
```

```python
import transformers
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)
```

### Accelerate
```bash
pip install accelerate
```

```python
import accelerate
```

---

## 2. LoRA / Fine-tuning

### PEFT
```bash
pip install peft
```

```python
import peft
from peft import LoraConfig
```

### Safetensors
```bash
pip install safetensors
```

```python
from safetensors.torch import load_file, save_file
```

### BitsAndBytes
```bash
pip install bitsandbytes
```

```python
import bitsandbytes
```

---

## 3. Dataset

### Hugging Face Datasets
```bash
pip install datasets
```

```python
from datasets import load_dataset, Dataset, DatasetDict
```

### Hugging Face Hub
```bash
pip install huggingface_hub
```

```python
from huggingface_hub import (
    login,
    hf_hub_download,
    snapshot_download,
)
```

---

## 4. Image Processing

### Pillow
```bash
pip install pillow
```

```python
from PIL import Image
```

### OpenCV
```bash
pip install opencv-python
```

```python
import cv2
```

### NumPy
```bash
pip install numpy
```

```python
import numpy as np
```

### SciPy
```bash
pip install scipy
```

```python
import scipy
```

---

## 5. Face Identity

### InsightFace
```bash
pip install insightface
```

```python
import insightface
```

### ONNX Runtime
```bash
pip install onnxruntime
```

```python
import onnxruntime
```

> `InsightFace` is used for face detection, face recognition, and face embeddings.

---

## 6. IP-Adapter

IP-Adapter is a model/repository integration rather than a package that should simply be added to the main `pip install` command.

Repository:

```text
https://github.com/tencent-ailab/IP-Adapter
```

Depending on the implementation, imports may look like:

```python
from diffusers import StableDiffusionPipeline
```

and the IP-Adapter weights are loaded through the Diffusers pipeline.

---

## 7. InstantID

InstantID is an alternative identity-preservation component.

Repository:

```text
https://github.com/InstantID/InstantID
```

It should be installed according to the selected InstantID implementation and checkpoint.

Do not install both IP-Adapter and InstantID blindly. Start with one identity-preservation method.

---

## 8. Training Utilities

### tqdm
```bash
pip install tqdm
```

```python
from tqdm.auto import tqdm
```

### Matplotlib
```bash
pip install matplotlib
```

```python
import matplotlib.pyplot as plt
```

### TensorBoard
```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter
```

### Weights & Biases — Optional
```bash
pip install wandb
```

```python
import wandb
```

---

# 9. Recommended Installation

Install the core environment:

```bash
pip install torch torchvision torchaudio

pip install diffusers transformers accelerate

pip install peft safetensors bitsandbytes

pip install datasets huggingface_hub

pip install pillow opencv-python numpy scipy

pip install insightface onnxruntime

pip install tqdm matplotlib tensorboard
```

Optional:

```bash
pip install wandb
```

---

# 10. Minimal Import Set

For the first Stable Diffusion + LoRA training prototype:

```python
import torch
import numpy as np

from PIL import Image
from tqdm.auto import tqdm

from datasets import load_dataset
from huggingface_hub import login

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)

from peft import LoraConfig
```

For face identity:

```python
import cv2
import insightface
import onnxruntime
```

For training monitoring:

```python
from torch.utils.tensorboard import SummaryWriter
```

---

# 11. Package Checklist

- [ ] torch
- [ ] torchvision
- [ ] torchaudio
- [ ] diffusers
- [ ] transformers
- [ ] accelerate
- [ ] peft
- [ ] safetensors
- [ ] bitsandbytes
- [ ] datasets
- [ ] huggingface_hub
- [ ] pillow
- [ ] opencv-python
- [ ] numpy
- [ ] scipy
- [ ] insightface
- [ ] onnxruntime
- [ ] tqdm
- [ ] matplotlib
- [ ] tensorboard
- [ ] wandb (optional)
