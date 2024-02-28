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
license: mit
extra_gated_prompt: "The collected information will help acquire a better knowledge of pyannote.audio userbase and help its maintainers improve it further. Though this pipeline uses MIT license and will always remain open-source, we will occasionnally email you about premium pipelines and paid services around pyannote."
extra_gated_fields:
  Company/university: text
  Website: text
---

Using this open-source pipeline in production?  
Make the most of it thanks to our [consulting services](https://herve.niderb.fr/consulting.html).

# 🎹 Speaker diarization 3.0

This pipeline has been trained by Séverin Baroudi with [pyannote.audio](https://github.com/pyannote/pyannote-audio) `3.0.0` using a combination of the training sets of AISHELL, AliMeeting, AMI, AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE, and VoxConverse.

It ingests mono audio sampled at 16kHz and outputs speaker diarization as an [`Annotation`](http://pyannote.github.io/pyannote-core/structure.html#annotation) instance: 

* stereo or multi-channel audio files are automatically downmixed to mono by averaging the channels. 
* audio files sampled at a different rate are resampled to 16kHz automatically upon loading.


## Requirements

1. Install [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) `3.0` with `pip install pyannote.audio`
2. Accept [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) user conditions
3. Accept [`pyannote/speaker-diarization-3.0`](https://hf.co/pyannote-speaker-diarization-3.0) user conditions
4. Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

## Usage

```python
# instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.0",
  use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# run the pipeline on an audio file
diarization = pipeline("audio.wav")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
```

### Processing on GPU

`pyannote.audio` pipelines run on CPU by default.
You can send them to GPU with the following lines:

```python 
import torch
pipeline.to(torch.device("cuda"))
```

Real-time factor is around 2.5% using one Nvidia Tesla V100 SXM2 GPU (for the neural inference part) and one Intel Cascade Lake 6248 CPU (for the clustering part).

In other words, it takes approximately 1.5 minutes to process a one hour conversation.

### Processing from memory

Pre-loading audio files in memory may result in faster processing:

```python
waveform, sample_rate = torchaudio.load("audio.wav")
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```

### Monitoring progress

Hooks are available to monitor the progress of the pipeline:

```python
from pyannote.audio.pipelines.utils.hook import ProgressHook
with ProgressHook() as hook:
    diarization = pipeline("audio.wav", hook=hook)
```

### Controlling the number of speakers

In case the number of speakers is known in advance, one can use the `num_speakers` option:

```python
diarization = pipeline("audio.wav", num_speakers=2)
```

One can also provide lower and/or upper bounds on the number of speakers using `min_speakers` and `max_speakers` options:

```python
diarization = pipeline("audio.wav", min_speakers=2, max_speakers=5)
```

## Benchmark 

This pipeline has been benchmarked on a large collection of datasets.  

Processing is fully automatic:

* no manual voice activity detection (as is sometimes the case in the literature)
* no manual number of speakers (though it is possible to provide it to the pipeline)
* no fine-tuning of the internal models nor tuning of the pipeline hyper-parameters to each dataset

... with the least forgiving diarization error rate (DER) setup (named *"Full"* in [this paper](https://doi.org/10.1016/j.csl.2021.101254)):

* no forgiveness collar
* evaluation of overlapped speech

| Benchmark                                                                                                                                   | [DER%](. "Diarization error rate") | [FA%](. "False alarm rate") | [Miss%](. "Missed detection rate") | [Conf%](. "Speaker confusion rate") | Expected output                                                                                                               | File-level evaluation                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | --------------------------- | ---------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [AISHELL-4](http://www.openslr.org/111/)                                                                                                    | 12.3                              | 3.8                        | 4.4                               | 4.1                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AISHELL.SpeakerDiarization.Benchmark.test.rttm)          | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AISHELL.SpeakerDiarization.Benchmark.test.eval)          |
| [AliMeeting (*channel 1*)](https://www.openslr.org/119/)                                                                                    | 24.3                              | 4.4                        | 10.0                              | 9.9                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AliMeeting.SpeakerDiarization.Benchmark.test.rttm)       | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AliMeeting.SpeakerDiarization.Benchmark.test.eval)       |
| [AMI (*headset mix,*](https://groups.inf.ed.ac.uk/ami/corpus/) [*only_words*)](https://github.com/BUTSpeechFIT/AMI-diarization-setup)       | 19.0                              | 3.6                        | 9.5                               | 5.9                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AMI.SpeakerDiarization.Benchmark.test.rttm)              | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AMI.SpeakerDiarization.Benchmark.test.eval)              |
| [AMI (*array1, channel 1,*](https://groups.inf.ed.ac.uk/ami/corpus/) [*only_words)*](https://github.com/BUTSpeechFIT/AMI-diarization-setup) | 22.2                              | 3.8                        | 11.2                              | 7.3                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AMI-SDM.SpeakerDiarization.Benchmark.test.rttm)          | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AMI-SDM.SpeakerDiarization.Benchmark.test.eval)          |
| [AVA-AVD](https://arxiv.org/abs/2111.14448) |  49.1 | 10.8 | 15.7| 22.5 | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AVA-AVD.SpeakerDiarization.Benchmark.test.rttm)       | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/AVA-AVD.SpeakerDiarization.Benchmark.test.eval)       |
| [DIHARD 3 (*Full*)](https://arxiv.org/abs/2012.01477)                                                                                       | 21.7                              | 6.2                       | 8.1                               | 7.3                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/DIHARD.SpeakerDiarization.Benchmark.test.rttm)           | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/DIHARD.SpeakerDiarization.Benchmark.test.eval)           |
| [MSDWild](https://x-lance.github.io/MSDWILD/) | 24.6 | 5.8 | 8.0 | 10.7 | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/MSDWILD.SpeakerDiarization.Benchmark.test.rttm)       | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/MSDWILD.SpeakerDiarization.Benchmark.test.eval)       |
| [REPERE (*phase 2*)](https://islrn.org/resources/360-758-359-485-0/)                                                                        | 7.8                               | 1.8                        | 2.6                               | 3.5                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/REPERE.SpeakerDiarization.Benchmark.test.rttm)           | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/REPERE.SpeakerDiarization.Benchmark.test.eval)           |
| [VoxConverse (*v0.3*)](https://github.com/joonson/voxconverse)                                                                              | 11.3                              | 4.1                        | 3.4                               | 3.8                                | [RTTM](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/VoxConverse.SpeakerDiarization.Benchmark.test.rttm)       | [eval](https://huggingface.co/pyannote/speaker-diarization-3.0.0/blob/main/reproducible_research/VoxConverse.SpeakerDiarization.Benchmark.test.eval)       |


## Citations

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

```bibtex
@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```
