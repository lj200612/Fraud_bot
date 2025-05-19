import torch
import numpy as np
from typing import Union, List

def to_device(data: Union[torch.Tensor, List[torch.Tensor]], device: torch.device):
    """将数据移动到指定设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

def pad_sequence(sequences: List[np.ndarray], max_len: int = None) -> np.ndarray:
    """填充序列到相同长度"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_sequences = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    return padded_sequences

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """归一化音频数据"""
    return audio / np.max(np.abs(audio))

def format_entity(entity: dict) -> dict:
    """格式化实体信息"""
    return {
        "text": entity["text"],
        "type": entity["type"],
        "start": entity["start"],
        "end": entity["start"] + len(entity["text"])
    } 