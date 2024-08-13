from __future__ import annotations

from io import BytesIO

from speech_recognition.audio import AudioData
from speech_recognition.exceptions import SetupError


def recognize_faster_whisper(
    recognizer,
    audio_data,
    model="large-v3",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "default",
    download_root: str = None,
    show_dict: bool = False,
    **transcribe_options,
):
    if not isinstance(audio_data, AudioData):
        raise ValueError("``audio_data`` must be an ``AudioData`` instance")

    try:
        from faster_whisper import WhisperModel

        if (
            not hasattr(recognizer, "faster_whisper_model")
            or recognizer.faster_whisper_model.get(model) is None
        ):
            recognizer.faster_whisper_model = getattr(
                recognizer, "faster_whisper_model", {}
            )
            recognizer.faster_whisper_model[model] = WhisperModel(
                model,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )

    except ImportError:
        raise SetupError(
            "missing faster_whisper module: ensure that faster_whisper is set up correctly."
        )

    wav_stream = BytesIO(audio_data.get_wav_data(convert_rate=16000))

    segments, info = recognizer.faster_whisper_model[model].transcribe(
        wav_stream, language=language, beam_size=5, **transcribe_options
    )

    text = " ".join([segment.text for segment in segments])
    if show_dict:
        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }
    else:
        return text
