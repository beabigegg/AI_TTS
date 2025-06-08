# scripts/segment_audio_optimized.py
import pandas as pd
from pathlib import Path
import subprocess
import argparse
from tqdm import tqdm
import sys
import signal
import concurrent.futures
import os

# --- 全域變數和信號處理函式保持不變 ---
# 注意：在並行處理中，從信號處理器安全地寫入檔案比較複雜。
# 一個更穩健的方法可能是讓主進程定期從結果佇列中儲存進度，
# 但為了與原程式碼盡量相似，我們暫時保留它，儘管在並行化後其效果可能有限。
g_new_metadata_list = []
g_output_segments_dir = None

# (您的 save_current_progress 函式可以保留，但在並行模式下可能不如預期)
# 在 ProcessPoolExecutor 中，子進程不會繼承主進程的 signal handler。
# 更穩健的中斷處理需要更複雜的進程間通訊。

def process_segment(args_tuple):
    """
    處理單一音訊片段的函式，設計為可在獨立進程中運行。
    它接收一個元組作為參數以方便 ProcessPoolExecutor 的 map 方法。
    """
    (index, row, original_audio_path, output_segments_dir, 
     target_sample_rate, target_channels, original_audio_stem) = args_tuple

    text = row["text"]
    start_time = row["start"]
    end_time = row["end"]
    duration = float(end_time) - float(start_time)

    if not isinstance(text, str):
        text = ""

    if duration <= 0.5:
        # 返回一個狀態，表示因時長被跳過
        return {"status": "skipped_duration", "index": index}

    segment_filename = f"{original_audio_stem}_seg_{index:05d}.wav"
    output_segment_path = output_segments_dir / segment_filename
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_audio_path),
        "-ss", str(start_time),
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        "-ar", str(target_sample_rate),
        "-ac", str(target_channels),
        "-loglevel", "error",
        str(output_segment_path)
    ]
    
    try:
        # 使用 subprocess.run，它會等待命令完成
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        # 成功，返回新的 metadata
        return {
            "status": "success",
            "data": {
                "segment_filename": segment_filename,
                "text": text
            }
        }
    except subprocess.CalledProcessError as e:
        # FFmpeg 出錯，返回錯誤資訊
        error_message = f"FFmpeg 處理片段 (原始索引 {index}, 新檔名 {segment_filename}) 時發生錯誤: {e.stderr.strip()}"
        return {"status": "error", "message": error_message}
    except Exception as e_gen:
        error_message = f"處理片段 (原始索引 {index}, 新檔名 {segment_filename}) 時發生未知錯誤: {e_gen}"
        return {"status": "error", "message": error_message}


def segment_audio_ffmpeg_parallel(metadata_path: Path,
                                  raw_audio_parent_dir: Path,
                                  output_segments_dir_param: Path,
                                  target_sample_rate: int = 22050,
                                  target_channels: int = 1,
                                  max_workers: int = None) -> Path | None:
    
    global g_output_segments_dir
    g_output_segments_dir = output_segments_dir_param
    g_output_segments_dir.mkdir(parents=True, exist_ok=True)

    # 讀取 Metadata
    try:
        main_metadata_df = pd.read_csv(metadata_path)
        if main_metadata_df.empty:
            # 處理空檔案的情況...
            output_metadata_file = g_output_segments_dir / "metadata.csv"
            output_metadata_file.write_text("segment_filename,text\n", encoding='utf-8')
            print(f"輸入的 Metadata 檔案為空，已創建一個空的 metadata.csv。")
            return output_metadata_file
    except Exception as e:
        print(f"錯誤：讀取 Metadata 檔案 {metadata_path} 失敗: {e}")
        return None

    print(f"開始使用並行 ffmpeg 分割音訊片段至 {g_output_segments_dir}...")
    
    new_metadata_list = []
    skipped_due_to_duration_count = 0
    ffmpeg_error_count = 0

    # 確定並行任務的數量
    if max_workers is None:
        # os.cpu_count() 可以獲取 CPU 核心數，是一個很好的預設值
        max_workers = os.cpu_count() or 4
    print(f"將使用最多 {max_workers} 個並行進程。")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 遍歷每個大的音訊檔
        for original_filename_full, group in main_metadata_df.groupby("filename"):
            original_audio_path = raw_audio_parent_dir / original_filename_full
            original_audio_stem = original_audio_path.stem

            if not original_audio_path.exists():
                print(f"警告：找不到原始音訊檔案 {original_audio_path}，跳過相關片段。")
                continue

            print(f"\n正在處理檔案: {original_filename_full} (共 {len(group)} 個片段)")

            # 為這個檔案的所有片段準備任務
            tasks = []
            for index, row in group.iterrows():
                tasks.append((index, row, original_audio_path, g_output_segments_dir,
                              target_sample_rate, target_channels, original_audio_stem))

            # 將所有任務提交給 executor，並使用 tqdm 顯示進度
            # executor.map 會保持任務的順序
            results = list(tqdm(executor.map(process_segment, tasks), total=len(tasks), desc=f"分割 {original_filename_full[:30]}"))

            # 處理結果
            for res in results:
                if res['status'] == 'success':
                    new_metadata_list.append(res['data'])
                elif res['status'] == 'skipped_duration':
                    skipped_due_to_duration_count += 1
                elif res['status'] == 'error':
                    ffmpeg_error_count += 1
                    tqdm.write(res['message']) # 使用 tqdm.write 避免與進度條衝突

    print("\n所有檔案處理完成。")
    print(f"  成功處理的片段數: {len(new_metadata_list)}")
    print(f"  因時長過短跳過的片段數: {skipped_due_to_duration_count}")
    print(f"  因 FFmpeg 錯誤跳過的片段數: {ffmpeg_error_count}")

    # 儲存最終的 metadata.csv
    output_metadata_file = g_output_segments_dir / "metadata.csv"
    if new_metadata_list:
        new_metadata_df = pd.DataFrame(new_metadata_list)
        new_metadata_df.to_csv(output_metadata_file, index=False, encoding='utf-8')
        print(f"✅ 並行分割完成！新的 metadata.csv (包含 {len(new_metadata_df)} 條記錄) 已儲存至 {output_metadata_file}")
    else:
        pd.DataFrame(columns=["segment_filename", "text"]).to_csv(output_metadata_file, index=False, encoding='utf-8')
        print(f"警告：並行分割後沒有生成任何有效的片段記錄。已創建空的 metadata.csv。")
        
    return output_metadata_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 FFmpeg 並行分割音訊，並進行重採樣和單聲道轉換")
    # ... (您的 argparse 參數定義保持不變) ...
    parser.add_argument("--metadata_path", required=True, type=Path, help="...")
    parser.add_argument("--raw_audio_parent_dir", required=True, type=Path, help="...")
    parser.add_argument("--output_dir", required=True, type=Path, help="...")
    parser.add_argument("--target_sample_rate", type=int, default=22050, help="...")
    parser.add_argument("--target_channels", type=int, default=1, help="...")
    # 新增一個參數來控制並行數量
    parser.add_argument("--max_workers", type=int, default=None, help="最大並行進程數，預設為系統 CPU 核心數")
    args = parser.parse_args()

    # 調用新的並行函式
    result_metadata_path = segment_audio_ffmpeg_parallel(
        args.metadata_path, 
        args.raw_audio_parent_dir, 
        args.output_dir,
        args.target_sample_rate,
        args.target_channels,
        args.max_workers
    )

    if result_metadata_path is None:
        print("並行分割步驟執行失敗。")
        sys.exit(1)
    else:
        # (您的最終檢查程式碼可以保留)
        print("腳本執行完畢。")