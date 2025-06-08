# scripts/denoise_audio_final.py
#
# 功能:
# 1. 對音訊檔案進行去噪，提取人聲 (基於 Demucs 模型)。
# 2. 自動偵測模式：
#    - 長音訊模式：若輸入目錄無 metadata.csv，則採用循序處理，節省 VRAM，適合處理長檔案。
#    - 短片段模式：若輸入目錄有 metadata.csv，則採用高效的批次處理。
# 3. [核心優化] 在短片段模式下，會先計算所有片段長度並排序，
#    再進行批次處理，從根本上避免因片段長度差異導致的 VRAM 耗盡和效能崩潰問題。
# 4. 修正了 torch.amp.autocast 的 FutureWarning。

import argparse
from pathlib import Path
import torchaudio
import torch
import torch.nn.functional as F
from demucs.apply import apply_model
from demucs.pretrained import get_model as get_demucs_model
from demucs.audio import convert_audio
from tqdm import tqdm
import pandas as pd

# --- 全域設定 ---
TARGET_DEMUCS_MODEL = "htdemucs_ft"
TARGET_SR_FOR_OUTPUT = 22050 # 最終輸出給 TTS 模型的採樣率


def process_batch(batch_wavs, demucs_model, device):
    """
    (此函式僅用於短片段批次模式)
    對一個批次的音訊張量進行 Demucs 推論。
    - batch_wavs: 一個張量列表，每個張量代表一個音訊。
    """
    # 1. 獲取批次中最長音訊的長度
    max_len = max(wav.shape[-1] for wav in batch_wavs)
    
    # 2. 對批次中的每個音訊進行填充 (padding)，使其長度一致
    padded_batch = [F.pad(wav, (0, max_len - wav.shape[-1])) for wav in batch_wavs]
    
    # 3. 將列表堆疊成一個批次張量
    batch_tensor = torch.stack(padded_batch).to(device)
    
    # 預處理（正規化）
    ref = batch_tensor.mean(dim=1, keepdim=True)
    batch_norm = (batch_tensor - ref.mean(dim=-1, keepdim=True)) / ref.std(dim=-1, keepdim=True)
    
    # 4. 一次性對整個批次進行模型推論
    # 已修正 FutureWarning：使用 torch.amp.autocast
    with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
        all_sources = apply_model(demucs_model, batch_norm, device=device, progress=False, num_workers=4)
    
    # 提取人聲並反正規化
    vocals_idx = demucs_model.sources.index('vocals')
    vocals_batch = all_sources[:, vocals_idx]
    vocals_denorm = vocals_batch * ref.std(dim=-1, keepdim=True) + ref.mean(dim=-1, keepdim=True)
    
    return vocals_denorm

def denoise_audio_final(
    input_dir: Path, 
    output_dir: Path, 
    min_duration_sec: float, 
    batch_size: int,
    is_long_audio_mode: bool
):
    """
    統一的音訊去噪與過濾函式。
    - is_long_audio_mode=True: 使用循序模式處理長音訊，節省VRAM。
    - is_long_audio_mode=False: 使用按長度排序的批次模式處理短片段，最大化GPU效率。
    """
    # --- 1. 初始化與模型載入 (對兩種模式都通用) ---
    if not input_dir.is_dir():
        print(f"錯誤：輸入目錄 {input_dir} 不存在。")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"載入 Demucs 模型: {TARGET_DEMUCS_MODEL}...")
    demucs_model = get_demucs_model(name=TARGET_DEMUCS_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        print("警告：未檢測到 CUDA，將使用 CPU。處理速度會非常慢。")
    demucs_model.to(device)
    demucs_model.eval()
    print(f"Demucs 模型已載入至 {device}。")

    # --- 2. 根據模式選擇不同的處理路徑 ---
    if is_long_audio_mode:
        # --- 長音訊模式：循序處理，節省記憶體 ---
        print("\n執行長音訊模式（循序處理以節省 VRAM）...")
        audio_files = list(input_dir.glob("*.wav"))
        if not audio_files:
            print(f"警告：在 {input_dir} 中未找到 .wav 檔案。")
            return

        for audio_path in tqdm(audio_files, desc="長音訊降噪進度"):
            try:
                wav, sr = torchaudio.load(audio_path)
                wav = wav.to(device)
                if wav.shape[0] == 1: wav = wav.repeat(2, 1) # Demucs 需要立體聲
                
                # 直接對單個長檔案應用模型，demucs會內部處理分塊
                # 這裡 num_workers 設為 0 在某些環境下更穩定
                sources = apply_model(demucs_model, wav[None], device=device, progress=True, num_workers=0)[0]
                
                vocals_idx = demucs_model.sources.index('vocals')
                vocals = sources[vocals_idx]

                # 輸出時轉換為目標採樣率和單聲道
                vocals_mono = convert_audio(vocals, demucs_model.samplerate, TARGET_SR_FOR_OUTPUT, 1)

                output_file_path = output_dir / audio_path.name
                torchaudio.save(output_file_path, vocals_mono.cpu(), TARGET_SR_FOR_OUTPUT)
                
            except Exception as e:
                tqdm.write(f"處理檔案 {audio_path.name} 時發生錯誤: {e}")
                
    else:
        # --- 短片段模式：按長度排序 + 批次處理，最大化效率 ---
        print(f"\n執行短片段模式（批次大小: {batch_size}）...")
        input_metadata_path = input_dir / "metadata.csv"
        if not input_metadata_path.exists():
            print(f"錯誤：在短片段模式下，找不到 {input_metadata_path}。")
            return
        
        input_metadata_df = pd.read_csv(input_metadata_path)

        # --- [核心優化] 預計算長度並排序 ---
        print("步驟 1/3：正在預先計算所有音訊片段的長度...")
        durations = []
        for filename in tqdm(input_metadata_df['segment_filename'], desc="計算時長"):
            audio_path = input_dir / filename
            try:
                if audio_path.exists():
                    info = torchaudio.info(str(audio_path))
                    duration = info.num_frames / info.sample_rate
                    # 同時過濾過短的檔案
                    if duration >= min_duration_sec:
                        durations.append(duration)
                    else:
                        durations.append(-1) # 標記為過短
                else:
                    durations.append(0) # 標記為不存在
            except Exception:
                durations.append(-1) # 標記為損壞或過短

        input_metadata_df['duration'] = durations
        
        # 過濾掉不存在、損壞或過短的檔案
        original_count = len(input_metadata_df)
        sorted_metadata_df = input_metadata_df[input_metadata_df['duration'] > 0].copy()
        skipped_count = original_count - len(sorted_metadata_df)
        print(f"已過濾 {skipped_count} 個不存在、損壞或過短的片段。")
        
        # 按長度排序！
        print("步驟 2/3：根據音訊長度進行排序以優化批次處理...")
        sorted_metadata_df = sorted_metadata_df.sort_values(by='duration').reset_index(drop=True)
        
        print(f"步驟 3/3：開始對 {len(sorted_metadata_df)} 個有效片段進行批次降噪...")
        final_metadata_list = []
        processed_count = 0
        skipped_error_count = 0
        batch_buffer = []

        iterable = tqdm(sorted_metadata_df.iterrows(), total=sorted_metadata_df.shape[0], desc="已排序的短片段降噪")

        for index, row in iterable:
            segment_filename = row["segment_filename"]
            text_for_metadata = row["text"]
            audio_path = input_dir / segment_filename

            try:
                wav, sr = torchaudio.load(audio_path)
                
                # 預處理音訊（重採樣、轉立體聲）
                if sr != demucs_model.samplerate:
                    wav = torchaudio.functional.resample(wav, sr, demucs_model.samplerate)
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                
                batch_buffer.append((segment_filename, text_for_metadata, wav))
                
                is_last_item = index == sorted_metadata_df.shape[0] - 1
                if (len(batch_buffer) >= batch_size or (is_last_item and batch_buffer)):
                    filenames, texts, wavs = zip(*batch_buffer)
                    
                    processed_vocals = process_batch(wavs, demucs_model, device)

                    for i in range(len(filenames)):
                        vocals_tensor = processed_vocals[i][..., :wavs[i].shape[-1]]
                        vocals_mono = convert_audio(vocals_tensor, demucs_model.samplerate, TARGET_SR_FOR_OUTPUT, 1)
                        torchaudio.save(output_dir / filenames[i], vocals_mono.cpu(), TARGET_SR_FOR_OUTPUT)
                        
                        final_metadata_list.append({"filename": filenames[i], "text": texts[i]})
                        processed_count += 1
                    
                    batch_buffer = []

            except Exception as e:
                tqdm.write(f"處理檔案 {segment_filename} 時發生錯誤: {e}")
                skipped_error_count += 1

        # 儲存 metadata
        if final_metadata_list:
            pd.DataFrame(final_metadata_list).to_csv(output_dir / "metadata.csv", index=False, encoding='utf-8')
        
        print("\n短片段處理統計：")
        print(f"  成功處理並保留: {processed_count} 個")
        print(f"  在預計算階段已跳過: {skipped_count} 個")
        print(f"  在處理階段因錯誤跳過: {skipped_error_count} 個")

    print("\n所有處理已完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="對音訊進行去噪。自動偵測長音訊（循序處理）或短片段（高效批次處理）。")
    parser.add_argument("--input_dir", required=True, type=Path, 
                        help="包含音訊檔案的目錄。若內含 'metadata.csv'，則啟用短片段模式。")
    parser.add_argument("--output_dir", required=True, type=Path, 
                        help="儲存處理後音訊的目錄")
    parser.add_argument("--min_duration", type=float, default=0.5, 
                        help="[短片段模式] 最小音訊時長（秒），短於此值將在預計算階段被過濾。")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="[短片段模式] 批次大小，請根據您的 GPU VRAM 進行調整。")
    args = parser.parse_args()

    # --- 自動偵測模式 ---
    is_long_mode = not (args.input_dir / "metadata.csv").exists()
    
    if is_long_mode:
        print("偵測到長音訊模式 (輸入目錄中無 metadata.csv)。")
    else:
        print("偵測到短片段模式 (輸入目錄中找到 metadata.csv)。")

    denoise_audio_final(
        args.input_dir, 
        args.output_dir, 
        args.min_duration, 
        args.batch_size, 
        is_long_audio_mode=is_long_mode
    )