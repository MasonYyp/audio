---
tags:
- pyannote
- pyannote-audio
- pyannote-audio-model
- audio
- voice
- speech
- speaker
- speaker-segmentation
- voice-activity-detection
- overlapped-speech-detection
- resegmentation
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


# 🎹 Speaker segmentation

[Paper](http://arxiv.org/abs/2104.04045) | [Demo](https://huggingface.co/spaces/pyannote/pretrained-pipelines) | [Blog post](https://herve.niderb.fr/fastpages/2022/10/23/One-speaker-segmentation-model-to-rule-them-all)

![Example](example.png)

## Usage

Relies on pyannote.audio 2.1.1: see [installation instructions](https://github.com/pyannote/pyannote-audio).

```python
# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token="ACCESS_TOKEN_GOES_HERE")
```

### Voice activity detection

```python
from pyannote.audio.pipelines import VoiceActivityDetection
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)
vad = pipeline("audio.wav")
# `vad` is a pyannote.core.Annotation instance containing speech regions
```

### Overlapped speech detection

```python
from pyannote.audio.pipelines import OverlappedSpeechDetection
pipeline = OverlappedSpeechDetection(segmentation=model)
pipeline.instantiate(HYPER_PARAMETERS)
osd = pipeline("audio.wav")
# `osd` is a pyannote.core.Annotation instance containing overlapped speech regions
```

### Resegmentation

```python
from pyannote.audio.pipelines import Resegmentation
pipeline = Resegmentation(segmentation=model, 
                          diarization="baseline")
pipeline.instantiate(HYPER_PARAMETERS)
resegmented_baseline = pipeline({"audio": "audio.wav", "baseline": baseline})
# where `baseline` should be provided as a pyannote.core.Annotation instance
```

### Raw scores

```python
from pyannote.audio import Inference
inference = Inference(model)
segmentation = inference("audio.wav")
# `segmentation` is a pyannote.core.SlidingWindowFeature
# instance containing raw segmentation scores like the 
# one pictured above (output)
```


## Citation

```bibtex
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Address = {Brno, Czech Republic},
  Month = {August},
  Year = {2021},
```

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

## Reproducible research 

In order to reproduce the results of the paper ["End-to-end speaker segmentation for overlap-aware resegmentation
"](https://arxiv.org/abs/2104.04045), use `pyannote/segmentation@Interspeech2021` with the following hyper-parameters:

| Voice activity detection | `onset` | `offset` | `min_duration_on` | `min_duration_off` |
| ------------------------ | ------- | -------- | ----------------- | ------------------ |
| AMI Mix-Headset          | 0.684   | 0.577    | 0.181             | 0.037              |
| DIHARD3                  | 0.767   | 0.377    | 0.136             | 0.067              |
| VoxConverse              | 0.767   | 0.713    | 0.182             | 0.501              |

| Overlapped speech detection | `onset` | `offset` | `min_duration_on` | `min_duration_off` |
| --------------------------- | ------- | -------- | ----------------- | ------------------ |
| AMI Mix-Headset             | 0.448   | 0.362    | 0.116             | 0.187              |
| DIHARD3                     | 0.430   | 0.320    | 0.091             | 0.144              |
| VoxConverse                 | 0.587   | 0.426    | 0.337             | 0.112              |

| Resegmentation of VBx | `onset` | `offset` | `min_duration_on` | `min_duration_off` |
| --------------------- | ------- | -------- | ----------------- | ------------------ |
| AMI Mix-Headset       | 0.542   | 0.527    | 0.044             | 0.705              |
| DIHARD3               | 0.592   | 0.489    | 0.163             | 0.182              |
| VoxConverse           | 0.537   | 0.724    | 0.410             | 0.563              |

Expected outputs (and VBx baseline) are also provided in the `/reproducible_research` sub-directories.

