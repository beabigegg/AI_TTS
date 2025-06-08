# scripts/synthesize.py
# (原 06_synthesize.py，已使用 torch.hub 載入 WaveGlow)
# 用法:
# python -m scripts.synthesize --text "こんにちは" --output_path "outputs/synthesis/single.wav" --model_path "outputs/models/your_model.pth"
# python -m scripts.synthesize --text_file "my_script.txt" --output_path "outputs/synthesis/batch_combined.wav" --model_path "outputs/models/your_model.pth"

import torch
import torchaudio
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# 使用相對導入
from .common import TextToMelModel, text_to_sequence, _symbols

def synthesize_single_text(text: str, model: TextToMelModel, vocoder, device: str) -> torch.Tensor | None:
    """接收單一文本，返回音訊波形張量 (Tensor)"""
    if not text.strip():
        # print("警告：輸入文本為空，跳過合成。")
        return None

    sequence_list = text_to_sequence(text)
    if not sequence_list:
        # print(f"警告：文本 '{text[:30]}...' 轉換為音素序列後為空，跳過合成。")
        return None
    sequence = torch.LongTensor(sequence_list).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mel_prediction = model(sequence) # 推論模式
        
        # WaveGlow 的推論方式
        waveform, *_ = vocoder.infer(mel_prediction) # infer 返回 (waveform, z)
        
    return waveform.cpu() # (1, T_audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從文本或檔案合成音訊 (使用 WaveGlow)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="要合成的單一(日文)句子")
    group.add_argument("--text_file", type=Path, help="包含多行待合成句子的 .txt 檔案路徑")
    
    parser.add_argument("--model_path", type=Path, required=True, help="訓練好的 TextToMel 模型的路徑 (.pth)")
    parser.add_argument("--output_path", type=Path, required=True, help="輸出的 .wav 檔案路徑。")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用 'cuda' 或 'cpu'")
    
    args = parser.parse_args()

    # --- 載入模型 ---
    print("Step 1: 載入 TTS 模型和 WaveGlow 聲碼器...")
    device = torch.device(args.device)

    if not args.model_path.exists():
        print(f"錯誤：找不到 TTS 模型檔案 {args.model_path}")
        sys.exit(1)

    try:
        tts_model = TextToMelModel(n_vocab=len(_symbols), n_mels=80) # 假設 n_mels=80
        tts_model.load_state_dict(torch.load(args.model_path, map_location=device))
        tts_model.to(device).eval()
    except Exception as e:
        print(f"載入 TTS 模型 {args.model_path} 失敗: {e}")
        sys.exit(1)

    try:
        print("正在從 NVIDIA NGC 載入 WaveGlow 聲碼器 (首次執行可能需要下載)...")
        # torch.hub.set_dir(str(Path.home() / ".cache" / "torch" / "hub")) # 可選：設定 hub 目錄
        vocoder = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16', pretrained=True, progress=True)
        vocoder = vocoder.remove_weightnorm(vocoder)
        vocoder = vocoder.to(device).eval()
    except Exception as e:
        print(f"載入 WaveGlow 聲碼器失敗: {e}")
        print("請確保網路連線正常，或檢查 torch.hub 快取。")
        sys.exit(1)
    
    vocoder_sample_rate = 22050 # WaveGlow 預設使用 22050Hz
    
    print(f"模型載入完成！將使用 {vocoder_sample_rate}Hz 採樣率進行合成。")
    print("-" * 50)
    
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    final_waveform = None

    if args.text:
        print(f"正在合成單一句子: \"{args.text}\"")
        waveform_single = synthesize_single_text(args.text, tts_model, vocoder, device)
        if waveform_single is not None:
            final_waveform = waveform_single
        
    elif args.text_file:
        if not args.text_file.exists():
            print(f"錯誤：找不到輸入文字檔案 {args.text_file}")
            sys.exit(1)

        print(f"正在從檔案 {args.text_file} 讀取句子並合併...")
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"讀取文字檔案 {args.text_file} 失敗: {e}")
            sys.exit(1)

        if not lines:
            print(f"文字檔案 {args.text_file} 為空或不包含有效行。")
            sys.exit(1)

        all_waveforms = []
        silence = torch.zeros(int(vocoder_sample_rate * 0.5)) # 0.5 秒靜音

        for line in tqdm(lines, desc="批次合成進度"):
            waveform_segment = synthesize_single_text(line, tts_model, vocoder, device)
            if waveform_segment is not None:
                all_waveforms.append(waveform_segment.squeeze(0)) # (T_audio)
                all_waveforms.append(silence)

        if all_waveforms:
            # 移除最後一個多餘的靜音
            if len(all_waveforms) > 1 and torch.equal(all_waveforms[-1], silence):
                all_waveforms.pop()
            final_waveform = torch.cat(all_waveforms).unsqueeze(0) # (1, T_audio_total)
        else:
            print("未能成功合成任何有效音訊片段。")

    # --- 儲存最終的音訊檔 ---
    if final_waveform is not None and final_waveform.numel() > 0 : # 確保 final_waveform 不是空的
        try:
            torchaudio.save(str(args.output_path), final_waveform, sample_rate=vocoder_sample_rate)
            print("-" * 50)
            print(f"✅ 合成成功！音訊已儲存至: {args.output_path}")
            print("-" * 50)
        except Exception as e:
            print(f"儲存音訊檔案 {args.output_path} 失敗: {e}")
    else:
        print("沒有生成有效的音訊波形，無法儲存。")