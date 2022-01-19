import audio as Audio
import librosa
import numpy as np
import yaml
from tqdm import tqdm
import os

class Preprocessor:
    def __init__(self, config):
        self.sampling_rate = config["audio"]["sampling_rate"]
        self.STFT = Audio.stft.TacotronSTFT(
            config["stft"]["filter_length"],
            config["stft"]["hop_length"],
            config["stft"]["win_length"],
            config["mel"]["n_mel_channels"],
            config["audio"]["sampling_rate"],
            config["mel"]["mel_fmin"],
            config["mel"]["mel_fmax"],
        )

    def get_mel_from_file(self, in_path, meta_file, out_path):
        f = open(meta_file, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        outf = open(out_path + 'metadata.txt', 'w', encoding='utf-8')
        for line in tqdm(lines):
            line = line.split('|')
            filepath = in_path + line[0] + '.wav'
            if os.path.exists(filepath):
                #mel = self.get_single_mel(filepath)
                #np.save(out_path+'mels/'+line[0]+'.npy', mel, allow_pickle=False)
                outf.writelines(out_path+'mels/'+line[0]+'.npy|'+line[2])
            
        outf.close()


    def get_single_mel(self, in_wav_path):
        wav, _ = librosa.load(in_wav_path, self.sampling_rate)
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        
        return mel_spectrogram.T
        

if __name__ == "__main__":
    config = yaml.load(open('./hifigan/config.yaml', "r"), Loader=yaml.FullLoader)
    
    out_path = '/ceph/home/wxc20/tmpmy/tacotron/preprocessed_data/LJSpeech/'

    preprocessor = Preprocessor(config)
    preprocessor.get_mel_from_file('/ceph/datasets/LJSpeech-1.1/wavs/', '/ceph/datasets/LJSpeech-1.1/metadata.csv.txt', out_path)
    