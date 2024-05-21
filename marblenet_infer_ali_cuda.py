# Adapted from far field vad DH inference file
# Structured from https://github.com/Anwarvic/VAD_Benchmark/blob/main/vads/marblenet.py
import torch
from nemo.collections.asr.models import EncDecClassificationModel
import torchaudio
import numpy as np
import glob
import pandas as pd
import os
import time

import soundfile as sf
from sklearn.metrics import roc_auc_score

from vad import Vad
from utils import (
    convert_byte_to_tensor,
    convert_tensor_to_bytes,
    convert_tensor_to_numpy,
)

class MarbleNet(Vad):
    def __init__(self,
            threshold: float = 0.5, # 0.6, 0.7, 0.8 done
            window_size_ms: int = 100,
            step_size_ms: int = 100,
            model_name: str = "vad_marblenet",
            use_onnx: bool = False,
            use_cuda=True
        ):
        super().__init__(threshold, window_size_ms)
        # load vad model
        self._vad = self._load_pretrained_model()
        # use ONNX runtime for faster inference
        if use_onnx:
            import onnxruntime
            from tempfile import TemporaryDirectory
            # export the model to a tmp directory to be used by ONNX
            with TemporaryDirectory() as temp_dir:
                tmp_filepath = f"{temp_dir}/{model_name}.onnx"
                self._vad.export(tmp_filepath)
                self._onnx_session = onnxruntime.InferenceSession(tmp_filepath)
        else:
            self._onnx_session = None
        self._threshold = threshold
        self._window_size_ms = window_size_ms
        self._step_size_ms = step_size_ms
        self._valid_sr = [self._vad._cfg.sample_rate]
    

    def _load_pretrained_model(self):
        # Load pre-trained model
        vad = EncDecClassificationModel.restore_from(restore_path="./MarbleNet-3x2x64.nemo").to('cuda')
        # set model to inference mode
        vad.eval()
        return vad

    def infer_file(self, file_source, rttm_source, sampling_rate = 16000):
        chunk_ms = 100
        overlap = 0
        chunk_sample = int(sampling_rate * chunk_ms / 1000)
        step_sample = int(sampling_rate * chunk_ms * (1 - overlap) / 1000)

        ### 1. wav format
        audio, sr = torchaudio.load(file_source)
        audio = audio.to(self._vad.device)

        ### 2. flac format
        # audio, sr = sf.read(file_source)
        # audio = torch.from_numpy(audio).float()
        # audio = audio.to(self._vad.device)
        # assert len(audio.shape) == 1, audio.shape
        # audio = audio.unsqueeze(0)

        len_audio = audio.shape[-1]

        if len_audio % chunk_sample > 0:
            len_pad = chunk_sample - len_audio % chunk_sample
            # pad_audio = torch.zeros(audio.shape[0], len_pad)
            pad_audio = torch.zeros_like(audio[:, :len_pad])
            audio = torch.cat([audio, pad_audio], dim=-1)

        if sampling_rate != sr:
            print(f'Resample from {sr} to {sampling_rate}')
            audio = torchaudio.functional.resample(audio.unsqueeze(0), orig_freq=sr, new_freq=sampling_rate).squeeze(0) # squeeze and unsqueeze added to make it work on cuda

        # convert to single channel
        if len(audio.shape) > 1:
            audio = audio[0, :]

        vad_mask = torch.zeros(audio.shape)
    
        small_noise = torch.randn(audio.shape) * 1e-6

        t0 = time.time()

        for st_sample in range(0, len(audio), step_sample):
            audio_chunk = audio[st_sample:st_sample+chunk_sample]
            
            audio_signal_length = (torch.as_tensor(audio_chunk.size()).to(self._vad.device))
            audio_chunk = audio_chunk.unsqueeze(dim=0).to(self._vad.device)
            logits = self._vad.forward(
                input_signal=audio_chunk,
                input_signal_length=audio_signal_length
                )[0]

            probs = torch.softmax(logits, dim=-1)
            speech_prob = probs[1].item()
            speech_prob = 1 if speech_prob >= self._threshold else 0 

            sample_mask = torch.full((chunk_sample,),speech_prob)
            sample_mask = sample_mask.to(self._vad.device)
            vad_mask = vad_mask.to(self._vad.device) 
            vad_mask[..., st_sample:st_sample+chunk_sample] += sample_mask

        t = time.time() - t0
            
        vad_mask = torch.where(vad_mask > 1, 1, vad_mask)
        small_noise = small_noise.to(self._vad.device)
        masked_audio = vad_mask * audio + (1 - vad_mask) * small_noise

        gt_mask = torch.zeros_like(vad_mask)
        gt_mask = gt_mask.to(self._vad.device)
        utt_id = file_source.strip().split('/')[-1].split('.')[0]
        
        f_rttm = open(rttm_source, 'r')
        for line in f_rttm.readlines():
            tokens = line.strip().split()
            start, end = int(float(tokens[3]) * sr), int((float(tokens[3]) + float(tokens[4])) * sr)
            assert end <= gt_mask.shape[0], (end, gt_mask.shape)
            gt_mask[start-1: end] = 1

        acc = (vad_mask == gt_mask).sum() / gt_mask.shape[0]
        fa = ((vad_mask == 1) & (gt_mask == 0)).sum() / gt_mask.shape[0]
        missing = ((vad_mask == 0) & (gt_mask == 1)).sum() / gt_mask.shape[0]
        roc_auc = roc_auc_score(gt_mask.cpu().numpy(), vad_mask.cpu().numpy()) #can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first

        print(f'{utt_id}: Acc = {acc:.3f}, False alarm = {fa:.3f}, Missing detection = {missing:.3f}, ROC-AUC score = {roc_auc:.3f}')

        return acc, fa, missing, roc_auc, t, gt_mask.shape[-1]/16000

if __name__ == "__main__":
    from os.path import dirname, abspath, join

    print("Running MarbleNet Vad")
    vad = MarbleNet()
    # samples_dir = "/home/users/ntu/kshitij0/scratch/datasets/alimeeting/Test_Ali/Test_Ali_far/audio_dir"
    # rttm_dir = "/home/users/ntu/kshitij0/scratch/datasets/alimeeting/Test_Ali/Test_Ali_far/rttm_dir"
    samples_dir = "/home/users/ntu/kshitij0/scratch/datasets/Chunlei/wav"
    rttm_dir = "/home/users/ntu/kshitij0/scratch/datasets/Chunlei/rttm"

    audio_files = glob.glob(join(samples_dir, "*.wav"))
    rttm_files = glob.glob(join(rttm_dir, "*.rttm"))

    print(f"Found {len(audio_files)} audio files in {samples_dir}")
    print(f"Found {len(rttm_files)} audio files in {rttm_dir}")

    sorted_audio_files = sorted(audio_files)
    sorted_rttm_files = sorted(rttm_files)
    
    total_acc, total_fa, total_missing, total_roc_auc, num_samples = 0, 0, 0, 0, 0
    total_t1, total_t2 = 0, 0 

    if len(sorted_audio_files) == len(sorted_rttm_files):
        for audio_file, rttm_file in zip(sorted_audio_files, sorted_rttm_files):
            # print(f"Processing file: {audio_file}")
            # file_name = os.path.splitext(os.path.basename(audio_file))[0]
            
            acc, fa, missing, roc_auc, t1, t2 = vad.infer_file(audio_file, rttm_file)
            print(f'processing time: {t1:.3f}, speech duration: {t2:.3f}')
            total_acc += acc
            total_fa += fa
            total_missing += missing
            total_roc_auc += roc_auc
            num_samples += 1
            total_t1 += t1
            total_t2 += t2
        
        total_acc /= num_samples
        total_fa /= num_samples
        total_missing /= num_samples
        total_roc_auc /= num_samples

        print("Done processing all files!")
        print('======================== Results ========================')
        print(f'Acc = {total_acc:.3f}, Missing detection = {total_missing:.3f}, False alarm = {total_fa:.3f}, ROC-AUC score = {total_roc_auc:.3f}')
        print(f'Processing time = {total_t1:.3f}, speech duration = {total_t2:.3f}, RTF = {total_t2 / total_t1:.3f}')
        print('Inference completed!')

    else:
        print("len(sorted_audio_files) =/= len(sorted_rttm_files)")
        for i, (audio_file, rttm_file) in enumerate(zip(sorted_audio_files, sorted_rttm_files), 1):
            audio_name = os.path.splitext(os.path.basename(audio_file))[0]
            rttm_name = os.path.splitext(os.path.basename(rttm_file))[0]

            if audio_name != rttm_name:
                print(f"Iteration {i}: Names are not equal for {audio_file} and {rttm_file}")

    

    