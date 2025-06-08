# scripts/chat.py
# (原 07_interactive_chat.py，已使用 torch.hub 載入 WaveGlow 並修正 sample_rate)
# 用法: python -m scripts.chat [--model_path "path/to/your_model.pth"] [--llm_model "qwen:7b"]

import torch
import torchaudio
from pathlib import Path
import argparse
import ollama
import json
import simpleaudio as sa
import tempfile
import sys
import os

# 使用相對導入
from .common import TextToMelModel, text_to_sequence, _symbols

# --- TTS 合成函式 ---
def synthesize_speech_for_chat(text: str, model: TextToMelModel, vocoder, device: str, sample_rate: int) -> str | None:
    """
    接收文本，生成語音，並將其儲存為一個暫存 .wav 檔案。
    返回該暫存檔案的路徑，如果合成失敗則返回 None。
    """
    if not text.strip():
        print("[TTS 警告] 輸入文本為空，跳過合成。")
        return None

    sequence_list = text_to_sequence(text)
    if not sequence_list:
        print(f"[TTS 警告] 文本 '{text[:30]}...' 轉換為音素序列後為空，跳過合成。")
        return None
    sequence = torch.LongTensor(sequence_list).unsqueeze(0).to(device)
    
    with torch.no_grad():
        try:
            mel_prediction = model(sequence)
            waveform, *_ = vocoder.infer(mel_prediction) # WaveGlow
        except Exception as e:
            print(f"[TTS 錯誤] 生成梅爾頻譜或波形時失敗: {e}")
            return None

    try:
        temp_wav_path = tempfile.mktemp(suffix=".wav", prefix="chat_audio_", dir=Path("outputs")/ "temp_chat_audio")
        Path(temp_wav_path).parent.mkdir(parents=True, exist_ok=True) # 確保暫存目錄存在
        torchaudio.save(temp_wav_path, waveform.cpu(), sample_rate=sample_rate)
        return temp_wav_path
    except Exception as e:
        print(f"[TTS 錯誤] 儲存暫存音訊檔時失敗: {e}")
        return None

# --- 主聊天循環 ---
def main_chat_loop(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"將使用 {device} 裝置運行 TTS 模型。")

    # 1. 載入 TTS 模型和聲碼器
    print("正在載入 TTS 模型與 WaveGlow 聲碼器...")
    if not args.model_path.exists():
        print(f"錯誤：找不到 TTS 模型檔案 {args.model_path}")
        sys.exit(1)
    try:
        tts_model = TextToMelModel(n_vocab=len(_symbols), n_mels=80)
        tts_model.load_state_dict(torch.load(args.model_path, map_location=device))
        tts_model.to(device).eval()

        vocoder = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16', pretrained=True, progress=True)
        vocoder = vocoder.remove_weightnorm(vocoder)
        vocoder = vocoder.to(device).eval()
        vocoder_sample_rate = 22050 # WaveGlow 預設
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        sys.exit(1)
    
    print(f"TTS 引擎已準備就緒！(採樣率: {vocoder_sample_rate}Hz)")
    print("-" * 50)

    # 2. 設定 LLM
    system_prompt = """
    你是一個樂於助人的AI助理。
    請嚴格使用以下 JSON 格式回覆，不包含任何額外的解釋或文字：
    {
      "japanese": "這裡是用於語音輸出的日文回答",
      "chinese": "這裡是對應的中文翻譯，用於字幕"
    }
    如果使用者的問題不適合用日文回答，或者你無法生成有意義的日文回答，
    請在 "japanese" 欄位中填寫類似 "申し訳ありませんが、日本語での回答は難しいです。" 的句子。
    """
    messages = [{'role': 'system', 'content': system_prompt}]

    print(f"歡迎來到互動式聊天 (LLM: {args.llm_model})。輸入 'exit' 或 'quit' 來結束對話。")

    temp_audio_dir = Path("outputs") / "temp_chat_audio"
    temp_audio_dir.mkdir(parents=True, exist_ok=True)


    while True:
        try:
            user_input = input("您: ")
            if user_input.lower() in ['exit', 'quit', '再見', 'さようなら']:
                print("感謝使用，再見！")
                break
            if not user_input.strip():
                continue

            messages.append({'role': 'user', 'content': user_input})
            
            print("\nAI 思考中...")
            try:
                response = ollama.chat(model=args.llm_model, messages=messages, options={"temperature": 0.7})
                ai_response_content = response['message']['content']
                messages.append({'role': 'assistant', 'content': ai_response_content})
            except Exception as e:
                print(f"\n[LLM 錯誤] 與 Ollama 模型通訊失敗: {e}")
                messages.pop() # 移除失敗的使用者輸入，以免影響下次對話
                continue


            try:
                response_data = json.loads(ai_response_content)
                jp_text = response_data.get("japanese")
                cn_text = response_data.get("chinese")

                if not isinstance(jp_text, str) or not isinstance(cn_text, str): # 確保是字串
                    raise ValueError("JSON 回覆中的 'japanese' 或 'chinese' 鍵不是有效的字串。")

                print("-" * 50)
                print(f"字幕 (中文): {cn_text}")
                print("AI (日文): " + jp_text)
                
                if jp_text.strip(): # 只有當日文文本非空時才合成和播放
                    print("\n🔊 正在生成語音...")
                    wav_path = synthesize_speech_for_chat(jp_text, tts_model, vocoder, device, vocoder_sample_rate)
                    
                    if wav_path:
                        print("▶️ 正在播放聲音...")
                        try:
                            wave_obj = sa.WaveObject.from_wave_file(wav_path)
                            play_obj = wave_obj.play()
                            play_obj.wait_done()
                        except Exception as e_play:
                            print(f"[播放錯誤] 播放音訊 {wav_path} 失敗: {e_play}")
                        finally:
                            try: # 嘗試刪除暫存檔案
                                os.remove(wav_path)
                            except OSError:
                                pass # 如果刪除失敗，忽略
                    else:
                        print("[TTS] 未能生成有效語音。")
                else:
                    print("[TTS] 日文文本為空，不進行語音合成。")
                print("-" * 50)

            except (json.JSONDecodeError, ValueError) as e:
                print(f"\n[解析錯誤] 無法解析 AI 的回覆或回覆格式不正確: {e}")
                print(f"原始回覆內容: {ai_response_content}")
                print("-" * 50)

        except KeyboardInterrupt:
            print("\n偵測到中斷指令，再見！")
            break
        except Exception as e:
            print(f"\n發生未知錯誤: {e}")
            break
        finally:
            # 清理可能殘留的暫存音訊檔案
            for f_temp in temp_audio_dir.glob("chat_audio_*.wav"):
                try:
                    os.remove(f_temp)
                except OSError:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="與本地 LLM 和 TTS 模型進行互動式聊天")
    parser.add_argument("--model_path", type=Path, default=Path("outputs/models/text_to_mel_model.pth"), 
                        help="訓練好的 TextToMel 模型的路徑")
    parser.add_argument("--llm_model", type=str, default="qwen3:8b", 
                        help="要使用的 Ollama 模型名稱 (例如 qwen3:8b, llama3, etc.)")
    
    args = parser.parse_args()
    main_chat_loop(args)