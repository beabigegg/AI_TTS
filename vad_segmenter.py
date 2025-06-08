# scripts/vad_segmenter.py
# 功能: 使用 Silero VAD 對長音訊進行智能切割，移除靜音，生成適合 Whisper 處理的短片段。
# 用法: python scripts/vad_segmenter.py --input_dir "path/to/long_audios" --output_dir "path/to/vad_segments"

import torch
import torchaudio
from pathlib import Path
import argparse
from tqdm import tqdm

# VAD 模型和相關工具函數
def load_vad_model():
    """載入 Silero VAD 模型"""
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False)
        return model, utils
    except Exception as e:
        print(f"載入 Silero VAD 模型失敗: {e}")
        return None, None

def vad_segmenter(input_dir: Path, output_dir: Path, target_sr: int = 16000):
    """
    對輸入目錄中的所有 .wav 檔案進行 VAD 切割。
    """
    if not input_dir.is_dir():
        print(f"錯誤：輸入目錄 {input_dir} 不存在。")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    
    vad_model, (get_speech_timestamps, _, read_audio, *_) = load_vad_model()
    if not vad_model:
        return False
    
    audio_files = list(input_dir.glob("*.wav"))
    if not audio_files:
        print(f"在 {input_dir} 中沒有找到 .wav 檔案。")
        return False

    print(f"找到 {len(audio_files)} 個長音訊檔案，開始 VAD 智能切割...")

    for audio_path in tqdm(audio_files, desc="VAD 切割總進度"):
        try:
            # 使用 VAD 工具讀取音訊，它會自動重採樣至 16kHz
            wav_tensor = read_audio(str(audio_path), sampling_rate=target_sr)
            
            # 獲取語音時間戳 (以樣本數為單位)
            speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=target_sr)

            if not speech_timestamps:
                tqdm.write(f"警告：在 {audio_path.name} 中未偵測到語音活動，已跳過。")
                continue

            # 根據時間戳切割並儲存
            for i, ts in enumerate(speech_timestamps):
                start_sample = ts['start']
                end_sample = ts['end']
                
                chunk_tensor = wav_tensor[start_sample:end_sample]
                
                # 建立輸出檔名，格式：[原始檔名]_vad_chunk_[編號].wav
                output_filename = output_dir / f"{audio_path.stem}_vad_chunk_{i+1:04d}.wav"
                
                # 儲存切割後的音訊
                torchaudio.save(str(output_filename), chunk_tensor.unsqueeze(0), sample_rate=target_sr)

        except Exception as e:
            tqdm.write(f"處理檔案 {audio_path.name} 時發生 VAD 切割錯誤: {e}")

    print(f"\nVAD 智能切割完成！所有片段已儲存至: {output_dir}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Silero VAD 對長音訊進行智能切割")
    parser.add_argument("--input_dir", required=True, type=Path, help="包含長音訊 .wav 檔案的目錄")
    parser.add_argument("--output_dir", required=True, type=Path, help="儲存切割後語音片段的目錄")
    args = parser.parse_args()

    vad_segmenter(args.input_dir, args.output_dir)