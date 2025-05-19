import deepspeech
import numpy as np
import torch
import os
from tacotron2.hparams import create_hparams
from tacotron2.train import load_model
from tacotron2.text import text_to_sequence
import wave

class SpeechService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化DeepSpeech模型
        self.ds_model_path = os.path.join("models", "deepspeech")
        self.ds_model = deepspeech.Model(
            os.path.join(self.ds_model_path, "model.pbmm"),
            os.path.join(self.ds_model_path, "scorer.scorer")
        )
        
        # 初始化Tacotron2模型
        self.tacotron_path = os.path.join("models", "tacotron2")
        self.hparams = create_hparams()
        self.tacotron_model = load_model(self.hparams)
        self.tacotron_model.load_state_dict(
            torch.load(
                os.path.join(self.tacotron_path, "checkpoint_50000"),
                map_location=self.device
            )['state_dict']
        )
        self.tacotron_model.to(self.device)
        self.tacotron_model.eval()

    def speech_to_text(self, audio_file: bytes) -> str:
        """将语音转换为文本"""
        # 读取音频文件
        with wave.open(audio_file, 'rb') as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
        
        # 使用DeepSpeech进行语音识别
        text = self.ds_model.stt(audio)
        return text

    def text_to_speech(self, text: str) -> bytes:
        """将文本转换为语音"""
        # 文本预处理
        sequence = torch.tensor(
            text_to_sequence(text, ['chinese_cleaners']),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # 生成语音
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron_model.infer(sequence)
        
        # 将mel频谱图转换为音频
        audio = self.tacotron_model.decode(mel_outputs_postnet)
        
        # 将音频转换为字节流
        audio_bytes = audio.cpu().numpy().tobytes()
        return audio_bytes 