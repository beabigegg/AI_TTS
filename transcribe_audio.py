# scripts/transcribe_audio.py
# (原 02_transcribe.py)
# 用法: python scripts/transcribe_audio.py --input_dir "path/to/audio_files" --output_dir "path/to/transcripts"

import whisper_timestamped as whisper # 或者使用 openai-whisper
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
import torch

def transcribe_audio_series(input_dir: Path, output_dir: Path, language: str = "ja", model_size: str = "medium"):
    """
    對指定目錄下的所有 .wav 音訊檔進行轉錄，生成帶時間戳的 metadata.csv。
    """
    if not input_dir.is_dir():
        print(f"錯誤：輸入目錄 {input_dir} 不存在或不是一個目錄。")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"載入 Whisper 模型 ({model_size}) 至 {device}...")
    try:
        # 對於 whisper_timestamped
        model = whisper.load_model(model_size, device=device)
        # 如果使用 openai-whisper
        # model = whisper.load_model(model_size).to(device)
    except Exception as e:
        print(f"載入 Whisper 模型失敗: {e}")
        print("請確保已安裝 whisper 或 whisper_timestamped，並且模型名稱正確。")
        return False
    
    print("Whisper 模型載入完成。")

    audio_files = list(input_dir.glob("*.wav"))
    if not audio_files:
        print(f"在 {input_dir} 中沒有找到 .wav 檔案。")
        return False

    all_segments_data = []
    print(f"開始轉錄 {len(audio_files)} 個音訊檔案...")

    for audio_path in tqdm(audio_files, desc="Whisper 轉錄進度"):
        try:
            # 使用 whisper_timestamped
            result = whisper.transcribe(model, str(audio_path), language=language, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
            
            # 如果使用 openai-whisper
            # result = model.transcribe(str(audio_path), language=language, fp16=torch.cuda.is_available())

            for seg in result["segments"]:
                all_segments_data.append([
                    audio_path.name, # 記錄原始檔名
                    seg["text"].strip(),
                    round(seg["start"], 3),
                    round(seg["end"], 3)
                ])
        except Exception as e:
            tqdm.write(f"處理檔案 {audio_path.name} 轉錄時發生錯誤: {e}")

    if not all_segments_data:
        print("沒有成功轉錄任何片段。")
        return False

    df = pd.DataFrame(all_segments_data, columns=["filename", "text", "start", "end"])
    output_metadata_path = output_dir / "metadata.csv"
    df.to_csv(output_metadata_path, index=False, encoding='utf-8')
    print(f"轉錄完成！Metadata 已儲存至: {output_metadata_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Whisper 對音訊檔案進行轉錄並生成時間戳")
    parser.add_argument("--input_dir", required=True, type=Path, help="包含原始 .wav 音訊的目錄")
    parser.add_argument("--output_dir", required=True, type=Path, help="儲存轉錄結果 metadata.csv 的目錄")
    parser.add_argument("--language", type=str, default="ja", help="音訊的語言代碼 (例如: ja, en, zh)")
    parser.add_argument("--model_size", type=str, default="medium", help="Whisper 模型大小 (例如: tiny, base, small, medium, large)")
    args = parser.parse_args()

    transcribe_audio_series(args.input_dir, args.output_dir, args.language, args.model_size)