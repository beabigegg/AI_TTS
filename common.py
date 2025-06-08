# scripts/common.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyopenjtalk

# ---------------------
# 文本處理 (共用)
# ---------------------
_pad = "_"
_unk = "<UNK>" # 未知音素的特殊符號

# --- 您需要根據 "額外信息" 仔細更新此處的標點符號 ---
_punctuation = '、。，．・？！〽-…~《》%&\'(),-.0123456789:?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⓪' # 示例擴充，您需要更精確地從 "額外信息" 篩選
# 注意：上面我加入了一些常見的，但您務必從您的數據的 "額外信息" 中仔細挑選真正的標點。
# 避免將漢字、假名等內容字符放入 _punctuation。
# 數字和英文字母是否作為標點取決於您的處理邏輯，通常它們會被g2p處理或作為文本內容。
# 這裡我暫時將數字和英文全大寫字母放入，但這可能不是最佳實踐，取決於pyopenjtalk如何處理它們。
# 更好的做法可能是讓 pyopenjtalk 處理字母和數字，然後看它們是否被轉為音素。
# 如果它們沒有被轉為音素且您想讓模型識別它們，才考慮加入。

# 根據您的 extract_phonemes.py 輸出更新
_letters_chars = 'INUabdefghijkmnoprstuvwyz' # 不包含空格，因為 g2p().split() 後通常沒有獨立空格音素

_multichar_phonemes = ['by', 'ch', 'cl', 'gy', 'hy', 'ky', 'my', 'ny', 'pau', 'py', 'ry', 'sh', 'ts']

# _symbols 列表的構建
_symbols_list_elements = [_pad, _unk] + list(_punctuation) + list(_letters_chars) + _multichar_phonemes
_symbols = sorted(list(set(_symbols_list_elements))) # 排序並去重，確保定義正確

SYMBOL_TO_ID = {s: i for i, s in enumerate(_symbols)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(_symbols)}

UNK_ID = SYMBOL_TO_ID.get(_unk)
if UNK_ID is None: # 增加一個檢查確保 UNK_ID 被正確設置
    # 如果 _unk 在 _symbols_list_elements 中因為某些原因被去除了（例如 _punctuation 中包含了 "<UNK>"）
    # 這裡會出錯。確保 _unk 是獨特的。
    if _unk not in SYMBOL_TO_ID:
        _symbols.append(_unk) # 如果 UNK 被意外移除，嘗試重新添加
        _symbols = sorted(list(set(_symbols)))
        SYMBOL_TO_ID = {s: i for i, s in enumerate(_symbols)}
        ID_TO_SYMBOL = {i: s for i, s in enumerate(_symbols)}
        UNK_ID = SYMBOL_TO_ID.get(_unk)
        if UNK_ID is None:
             raise ValueError(f"CRITICAL: '{_unk}' symbol could not be added or found in SYMBOL_TO_ID. Check _symbols definition and ensure '{_unk}' is unique.")
    else: # 這種情況理論上不應發生，因為 .get() 會處理 KeyErorr
        UNK_ID = SYMBOL_TO_ID[_unk]


def text_to_sequence(text: str) -> list[int]:
    """將日語文本轉換為音素 ID 序列"""
    try:
        phonemes_str = pyopenjtalk.g2p(str(text), kana=False) # 確保 text 是字符串
    except Exception as e:
        # print(f"警告：pyopenjtalk 處理文本 '{str(text)[:30]}...' 時出錯: {e}，返回空序列。")
        return []

    sequence = []
    phonemes_list = phonemes_str.split(' ')

    for symbol in phonemes_list:
        if not symbol:
            continue
        # 直接查找分割後的音素單元
        token_id = SYMBOL_TO_ID.get(symbol)
        if token_id is not None:
            sequence.append(token_id)
        else:
            # 如果 symbol 不在 SYMBOL_TO_ID 中，但它可能是 _punctuation 中的字符
            # 或者它是一個應該被忽略的字符。
            # 這裡的邏輯是，如果 g2p 沒有把它轉成已知的音素，
            # 並且它也不是我們定義的標點或其他符號，就用 UNK。
            # pyopenjtalk 通常會處理標點，例如將其轉換為 "pau" 或移除。
            # 所以，如果一個 symbol 在這裡仍然未知，很可能它是 g2p 未能處理的罕見情況或新音素。
            # print(f"警告：在字典中找不到音素 '{symbol}' (來自文本: '{str(text)[:30]}...'), 使用 UNK 代替。")
            sequence.append(UNK_ID)
    return sequence

# ---------------------
# TTS 模型定義 (共用)
# (保持與您之前版本一致，這裡不再重複貼出 TextToMelModel 的完整程式碼)
# 確保 TextToMelModel 的 n_vocab 初始化時使用 len(_symbols)
# ---------------------
class TextToMelModel(nn.Module):
    """使用 Transformer Encoder 和插值來生成梅爾頻譜的模型"""
    def __init__(self, n_vocab, n_mels, hidden_dim=256, n_heads=4, num_encoder_layers=4, dim_feedforward_factor=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * dim_feedforward_factor,
            dropout=dropout,
            batch_first=True,
            activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.mel_out = nn.Linear(hidden_dim, n_mels)

    def forward(self, text_ids, mel_specs_target=None):
        padding_mask = (text_ids == SYMBOL_TO_ID[_pad]) # 假設 _pad 的 ID 也是通過 SYMBOL_TO_ID 獲取
        embedded_text = self.embedding(text_ids)
        encoder_output = self.transformer_encoder(embedded_text, src_key_padding_mask=padding_mask)

        mel_prediction = self.mel_out(encoder_output)
        mel_prediction = mel_prediction.transpose(1, 2) # (B, n_mels, T_text)

        if mel_specs_target is not None: # 訓練模式
            target_len = mel_specs_target.size(2)

            if mel_prediction.size(2) != target_len:
                mel_prediction_aligned = F.interpolate(mel_prediction, size=target_len, mode='linear', align_corners=False)
            else:
                mel_prediction_aligned = mel_prediction

            loss = F.l1_loss(mel_prediction_aligned, mel_specs_target)
            return loss

        return mel_prediction # 推論模式