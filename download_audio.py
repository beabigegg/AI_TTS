# scripts/download_audio.py
# (原 01_download_yt.py)
# 用法: python scripts/download_audio.py "YOUTUBE_URL" --series_name "MyAudioSeries" --output_base_dir "data/sources_raw"

import yt_dlp
from pathlib import Path
import os
import argparse

# load_dotenv() # 如果您使用 .env 檔案管理路徑，可以保留

def download_youtube(url: str, series_name: str, output_base_dir: Path):
    """下載YouTube音訊並保存到以 series_name 命名的子目錄中"""
    output_dir = output_base_dir / series_name
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        # 儲存到 output_dir 下，檔名為影片標題
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'extractaudio': True,
        'audioformat': 'wav', # 直接輸出 wav
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192', # 此品質參數對 wav 影響不大，但保留
        }],
        'quiet': False, # 設為 False 以便看到下載進度
        'ignoreerrors': True, # 遇到錯誤時繼續下載列表中的其他影片
        'nocheckcertificate': True, # 有時有用
    }

    print(f"開始下載音訊至: {output_dir}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"音訊已成功下載並儲存到: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"下載過程中發生錯誤: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從 YouTube 下載音訊")
    parser.add_argument("url", help="YouTube 影片或播放列表 URL")
    parser.add_argument("--series_name", required=True, help="此音訊系列的名稱 (將作為輸出子目錄名)")
    parser.add_argument("--output_base_dir", type=Path, default=Path("data/sources_raw"), help="儲存原始音訊的基底目錄")
    args = parser.parse_args()
    
    # 確保基底目錄存在
    args.output_base_dir.mkdir(parents=True, exist_ok=True)

    download_youtube(args.url, args.series_name, args.output_base_dir)