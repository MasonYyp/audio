---
tags: 
- pyannote
- pyannote-audio
- pyannote-audio-pipeline
- audio
- voice
- speech
- speaker
- speaker-diarization
- speaker-change-detection
- voice-activity-detection
- overlapped-speech-detection
- automatic-speech-recognition
datasets:
- ami
- dihard
- voxconverse
- aishell
- repere
- voxceleb
license: mit
extra_gated_prompt: "The collected information will help acquire a better knowledge of pyannote.audio userbase and help its maintainers apply for grants to improve it further. If you are an academic researcher, please cite the relevant papers in your own publications using the model. If you work for a company, please consider contributing back to pyannote.audio development (e.g. through unrestricted gifts). We also provide scientific consulting services around speaker diarization and machine listening."
extra_gated_fields:
  Company/university: text
  Website: text
  I plan to use this model for (task, type of audio data, etc): text
---

Using this open-source pipeline in production?  
Make the most of it thanks to our [consulting services](https://herve.niderb.fr/consulting.html).


# 🎹 Speaker diarization

Relies on pyannote.audio 2.1.1: see [installation instructions](https://github.com/pyannote/pyannote-audio#installation).

## TL;DR

```python
# 1. visit hf.co/pyannote/speaker-diarization and accept user conditions
# 2. visit hf.co/pyannote/segmentation and accept user conditions
# 3. visit hf.co/settings/tokens to create an access token
# 4. instantiate pretrained speaker diarization pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="ACCESS_TOKEN_GOES_HERE")


# apply the pipeline to an audio file
diarization = pipeline("audio.wav")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
```

## Advanced usage

In case the number of speakers is known in advance, one can use the `num_speakers` option:

```python
diarization = pipeline("audio.wav", num_speakers=2)
```

One can also provide lower and/or upper bounds on the number of speakers using `min_speakers` and `max_speakers` options:

```python
diarization = pipeline("audio.wav", min_speakers=2, max_speakers=5)
```

## Benchmark 

### Real-time factor

Real-time factor is around 2.5% using one Nvidia Tesla V100 SXM2 GPU (for the neural inference part) and one Intel Cascade Lake 6248 CPU (for the clustering part).

In other words, it takes approximately 1.5 minutes to process a one hour conversation.

### Accuracy

This pipeline is benchmarked on a growing collection of datasets.  

Processing is fully automatic:

* no manual voice activity detection (as is sometimes the case in the literature)
* no manual number of speakers (though it is possible to provide it to the pipeline)
* no fine-tuning of the internal models nor tuning of the pipeline hyper-parameters to each dataset

... with the least forgiving diarization error rate (DER) setup (named *"Full"* in [this paper](https://doi.org/10.1016/j.csl.2021.101254)):

* no forgiveness collar
* evaluation of overlapped speech


| Benchmark                                                                                                                                   | [DER%](. "Diarization error rate") | [FA%](. "False alarm rate") | [Miss%](. "Missed detection rate") | [Conf%](. "Speaker confusion rate") | Expected output                                                                                                               | File-level evaluation                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | --------------------------- | ---------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [AISHELL-4](http://www.openslr.org/111/)                                                                                                    | 14.09                              | 5.17                        | 3.27                               | 5.65                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/AISHELL.test.rttm)          | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/AISHELL.test.eval)          |
| [Albayzin (*RTVE 2022*)](http://catedrartve.unizar.es/albayzindatabases.html)                                                               | 25.60                              | 5.58                        | 6.84                               | 13.18                               | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/Albayzin2022.test.rttm)     | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/Albayzin2022.test.eval)     |
| [AliMeeting (*channel 1*)](https://www.openslr.org/119/)                                                                                    | 27.42                              | 4.84                        | 14.00                              | 8.58                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/AliMeeting.test.rttm)       | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/AliMeeting.test.eval)       |
| [AMI (*headset mix,*](https://groups.inf.ed.ac.uk/ami/corpus/) [*only_words*)](https://github.com/BUTSpeechFIT/AMI-diarization-setup)       | 18.91                              | 4.48                        | 9.51                               | 4.91                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/AMI.test.rttm)              | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/AMI.test.eval)              |
| [AMI (*array1, channel 1,*](https://groups.inf.ed.ac.uk/ami/corpus/) [*only_words)*](https://github.com/BUTSpeechFIT/AMI-diarization-setup) | 27.12                              | 4.11                        | 17.78                              | 5.23                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/AMI-SDM.test.rttm)          | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/AMI-SDM.test.eval)          |
| [CALLHOME](https://catalog.ldc.upenn.edu/LDC2001S97) [(*part2*)](https://github.com/BUTSpeechFIT/CALLHOME_sublists/issues/1)                | 32.37                              | 6.30                        | 13.72                              | 12.35                               | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/CALLHOME.test.rttm)         | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/CALLHOME.test.eval)         |
| [DIHARD 3 (*Full*)](https://arxiv.org/abs/2012.01477)                                                                                       | 26.94                              | 10.50                       | 8.41                               | 8.03                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/DIHARD.test.rttm)           | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/DIHARD.test.eval)           |
| [Ego4D *v1 (validation)*](https://arxiv.org/abs/2110.07058)                                                                                 | 63.99                              | 3.91                        | 44.42                              | 15.67                               | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/Ego4D.development.rttm)     | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/Ego4D.development.eval)     |
| [REPERE (*phase 2*)](https://islrn.org/resources/360-758-359-485-0/)                                                                        | 8.17                               | 2.23                        | 2.49                               | 3.45                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/REPERE.test.rttm)           | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/REPERE.test.eval)           |
| [This American Life](https://arxiv.org/abs/2005.08072)                                                                                      | 20.82                              | 2.03                        | 11.89                              | 6.90                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/ThisAmericanLife.test.rttm) | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/2.1.1/reproducible_research/2.1.1/ThisAmericanLife.test.eval) |
| [VoxConverse (*v0.3*)](https://github.com/joonson/voxconverse)                                                                              | 11.24                              | 4.42                        | 2.88                               | 3.94                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization/blob/main/reproducible_research/2.1.1/VoxConverse.test.rttm)       | [eval](https://huggingface.co/pyannote/speaker-diarization/blob/main/reproducible_research/2.1.1/VoxConverse.test.eval)       |

## Technical report

This [report](technical_report_2.1.pdf) describes the main principles behind version `2.1` of pyannote.audio speaker diarization pipeline.  
It also provides recipes explaining how to adapt the pipeline to your own set of annotated data. In particular, those are applied to the above benchmark and consistently leads to significant performance improvement over the above out-of-the-box performance.


## Citations

```bibtex
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Address = {Brno, Czech Republic},
  Month = {August},
  Year = {2021},
}
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
