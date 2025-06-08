import pyopenjtalk
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_phonemes_from_metadata(metadata_file_path: Path):
    """
    從 metadata.csv 檔案中讀取文本，使用 pyopenjtalk 轉換為音素，
    並分類為單字符和多字符音素。
    """
    if not metadata_file_path.exists():
        print(f"錯誤：找不到 Metadata 檔案 '{metadata_file_path}'")
        return

    try:
        df = pd.read_csv(metadata_file_path)
    except Exception as e:
        print(f"讀取 Metadata 檔案 '{metadata_file_path}' 失敗: {e}")
        return

    if 'text' not in df.columns:
        print(f"錯誤：Metadata 檔案 '{metadata_file_path}' 中缺少 'text' 欄位。")
        return

    all_phonemes_set = set()
    print(f"正在從 '{metadata_file_path}' 讀取文本並提取音素...")

    for text_entry in tqdm(df["text"], desc="處理文本"):
        if pd.isna(text_entry): # 跳過 NaN 值
            # tqdm.write(f"警告：檢測到空文本行，已跳過。") # 可選日誌
            continue
        try:
            # 將條目轉換為字符串，以防出現非字符串類型
            phonemes_str = pyopenjtalk.g2p(str(text_entry), kana=False)
            for p in phonemes_str.split(' '):
                if p:  # 確保不是空字符串
                    all_phonemes_set.add(p)
        except Exception as e:
            tqdm.write(f"警告：pyopenjtalk 處理文本 '{str(text_entry)[:30]}...' 時出錯: {e}")
            pass  # 忽略 g2p 可能的錯誤，繼續處理下一個

    print("\n--------------------------------------------------")
    print("所有偵測到的獨立音素單元 (未排序):")
    print(all_phonemes_set)
    print("--------------------------------------------------")

    # 為了方便定義 _letters_chars 和 _multichar_phonemes，進行分類
    single_char_phonemes = sorted(list(set(p for p in all_phonemes_set if len(p) == 1)))
    multi_char_phonemes = sorted(list(set(p for p in all_phonemes_set if len(p) > 1)))

    # 檢查是否有音素同時出現在單字符和多字符列表中（理論上不應該，除非定義有誤）
    # 但 pyopenjtalk 的輸出通常是明確的音素單元
    # 例如 'a' 和 'au' 是不同的

    print("\n【用於 common.py 的建議內容】")
    print("--------------------------------------------------")
    print("建議的 `_letters_chars` (單字符音素，請手動檢查並移除標點或非音素字符):")
    # 移除常見標點，因為它們由 _punctuation 處理
    # 您可能需要根據實際輸出手動調整這個過濾邏輯
    potential_letters = "".join(p for p in single_char_phonemes if p.isalnum() or p in ['↓', '↑']) # 簡單過濾
    print(f"_letters_chars = '{potential_letters}'")

    print("\n建議的 `_multichar_phonemes` (多字符音素列表):")
    print(f"_multichar_phonemes = {multi_char_phonemes}")
    print("--------------------------------------------------")

    # 額外：檢查文本中所有出現的字符，幫助識別標點符號
    all_text_chars_set = set()
    for text_entry in df["text"]:
        if pd.isna(text_entry):
            continue
        all_text_chars_set.update(list(str(text_entry)))

    print("\n【額外信息】")
    print("--------------------------------------------------")
    print("文本中出現的所有獨立字符 (用於檢查 `_punctuation` 是否完整):")
    # 過濾掉已知的音素字符，以便更容易看到標點符號
    # 這是一個粗略的過濾，可能不完美
    known_phoneme_chars = set(potential_letters)
    punctuation_candidates = sorted(list(c for c in all_text_chars_set if c not in known_phoneme_chars and not c.isspace()))
    print(punctuation_candidates)
    print("請將上述列表中的標點符號與您 common.py 中的 `_punctuation` 變數進行比較和更新。")
    print("--------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從 metadata.csv 提取並分類音素，以輔助 common.py 中的符號定義。")
    parser.add_argument("metadata_file", type=Path, help="metadata.csv 檔案的路徑。")
    args = parser.parse_args()

    extract_phonemes_from_metadata(args.metadata_file)