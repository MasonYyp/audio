---
tags:
- pyannote
- pyannote-audio
- pyannote-audio-model
- audio
- voice
- speech
- speaker
- speaker-recognition
- speaker-verification
- speaker-identification
- speaker-embedding
datasets:
- voxceleb
license: mit
inference: false
extra_gated_prompt: "The collected information will help acquire a better knowledge of pyannote.audio userbase and help its maintainers apply for grants to improve it further. If you are an academic researcher, please cite the relevant papers in your own publications using the model. If you work for a company, please consider contributing back to pyannote.audio development (e.g. through unrestricted gifts). We also provide scientific consulting services around speaker diarization and machine listening."
extra_gated_fields:
  Company/university: text
  Website: text
  I plan to use this model for (task, type of audio data, etc): text
---

Using this open-source model in production?  
Make the most of it thanks to our [consulting services](https://herve.niderb.fr/consulting.html).

# 🎹 Speaker embedding

Relies on pyannote.audio 2.1: see [installation instructions](https://github.com/pyannote/pyannote-audio/).

This model is based on the [canonical x-vector TDNN-based architecture](https://ieeexplore.ieee.org/abstract/document/8461375), but with filter banks replaced with [trainable SincNet features](https://ieeexplore.ieee.org/document/8639585). See [`XVectorSincNet`](https://github.com/pyannote/pyannote-audio/blob/3c988c028dc505c64fe776720372f6fe816b585a/pyannote/audio/models/embedding/xvector.py#L104-L169) architecture for implementation details.


## Basic usage

```python
# 1. visit hf.co/pyannote/embedding and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="ACCESS_TOKEN_GOES_HERE")
```

```python
from pyannote.audio import Inference
inference = Inference(model, window="whole")
embedding1 = inference("speaker1.wav")
embedding2 = inference("speaker2.wav")
# `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

from scipy.spatial.distance import cdist
distance = cdist(embedding1, embedding2, metric="cosine")[0,0]
# `distance` is a `float` describing how dissimilar speakers 1 and 2 are.
```

Using cosine distance directly, this model reaches 2.8% equal error rate (EER) on VoxCeleb 1 test set.   
This is without voice activity detection (VAD) nor probabilistic linear discriminant analysis (PLDA).
Expect even better results when adding one of those.

## Advanced usage

### Running on GPU

```python
import torch
inference.to(torch.device("cuda"))
embedding = inference("audio.wav")
```

### Extract embedding from an excerpt

```python
from pyannote.audio import Inference
from pyannote.core import Segment
inference = Inference(model, window="whole")
excerpt = Segment(13.37, 19.81)
embedding = inference.crop("audio.wav", excerpt)
# `embedding` is (1 x D) numpy array extracted from the file excerpt.
```

### Extract embeddings using a sliding window

```python
from pyannote.audio import Inference
inference = Inference(model, window="sliding",
                      duration=3.0, step=1.0)
embeddings = inference("audio.wav")
# `embeddings` is a (N x D) pyannote.core.SlidingWindowFeature
# `embeddings[i]` is the embedding of the ith position of the 
# sliding window, i.e. from [i * step, i * step + duration].
```


## Citation

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```

```bibtex
@inproceedings{Coria2020,
    author="Coria, Juan M. and Bredin, Herv{\'e} and Ghannay, Sahar and Rosset, Sophie",
    editor="Espinosa-Anke, Luis and Mart{\'i}n-Vide, Carlos and Spasi{\'{c}}, Irena",
    title="{A Comparison of Metric Learning Loss Functions for End-To-End Speaker Verification}",
    booktitle="Statistical Language and Speech Processing",
    year="2020",
    publisher="Springer International Publishing",
    pages="137--148",
    isbn="978-3-030-59430-5"
}
```

