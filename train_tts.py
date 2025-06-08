# scripts/train_tts.py
# (原 04_train.py)
# 用法: python -m scripts.train_tts --config configs/base_config.json --data_dir "path/to/final_training_data" [--output_model_path "outputs/models/my_custom_model.pth"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast

from pathlib import Path
import pandas as pd
import torchaudio
import logging
import json
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

# 從共用模組導入
from .common import TextToMelModel, SYMBOL_TO_ID, text_to_sequence, _symbols, _pad

# ---------------------
# 資料集定義
# ---------------------
class TextAudioDataset(Dataset):
    def __init__(self, metadata_path: Path, hps: dict):
        self.data_dir = metadata_path.parent
        try:
            self.df = pd.read_csv(metadata_path)
             # 確保 'filename' 和 'text' 欄位存在
            if 'filename' not in self.df.columns or 'text' not in self.df.columns:
                raise ValueError("Metadata CSV 必須包含 'filename' 和 'text' 欄位。")
        except Exception as e:
            logging.error(f"讀取 metadata 檔案 {metadata_path} 失敗: {e}")
            raise # 重新拋出錯誤，讓 DataLoader 知道此 Dataset 無法使用

        self.hps = hps
        self.n_fft = hps['data']['n_fft']
        self.skipped_files_count = 0

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=hps['data']['sampling_rate'],
            n_fft=self.n_fft,
            win_length=hps['data']['win_length'],
            hop_length=hps['data']['hop_length'],
            n_mels=hps['data']['n_mels']
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text = str(row["text"]) # 確保是字串
        filename = str(row["filename"]) # 確保是字串
        file_path = self.data_dir / filename

        try:
            audio, sr = torchaudio.load(file_path)
        except Exception as e:
            # logging.warning(f"讀取檔案 {file_path} 失敗: {e}，跳過。") # 在 tqdm 中會打亂顯示
            self.skipped_files_count += 1
            return None

        if audio.shape[-1] < self.n_fft:
            # logging.warning(f"音訊 {filename} 長度 ({audio.shape[-1]}) 過短 (n_fft {self.n_fft})，跳過。")
            self.skipped_files_count += 1
            return None

        if sr != self.hps['data']['sampling_rate']:
            audio = torchaudio.functional.resample(audio, sr, self.hps['data']['sampling_rate'])
        
        if audio.shape[0] > 1: # 如果是多聲道，轉換為單聲道
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        text_ids_list = text_to_sequence(text)
        if not text_ids_list: # 如果文本轉換後為空序列 (例如，只包含未知音素)
            # logging.warning(f"文本 '{text[:30]}...' (檔案 {filename}) 轉換為音素序列後為空，跳過。")
            self.skipped_files_count += 1
            return None
        text_ids = torch.LongTensor(text_ids_list)
        
        try:
            mel_spec = self.mel_spectrogram(audio)
        except Exception as e:
            # logging.warning(f"為檔案 {filename} 生成梅爾頻譜時失敗: {e}，跳過。")
            self.skipped_files_count += 1
            return None


        return {
            "text_ids": text_ids,
            "mel_spec": mel_spec.squeeze(0) # (n_mels, T_mel)
        }

# ---------------------
# Collate Function
# ---------------------
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    text_ids = [x["text_ids"] for x in batch]
    text_padded = pad_sequence(text_ids, batch_first=True, padding_value=SYMBOL_TO_ID[_pad])
    text_lengths = torch.LongTensor([len(x) for x in text_ids])

    # 梅爾頻譜的形狀是 (n_mels, T_mel)
    # 我們需要填充 T_mel 維度，所以先轉置成 (T_mel, n_mels)
    mel_specs_to_pad = [x["mel_spec"].permute(1, 0) for x in batch] # 每個元素是 (T_mel, n_mels)
    mel_padded = pad_sequence(mel_specs_to_pad, batch_first=True, padding_value=0.0) # (B, T_mel_max, n_mels)
    mel_padded = mel_padded.permute(0, 2, 1) # 轉回 (B, n_mels, T_mel_max)
    
    mel_lengths = torch.LongTensor([x["mel_spec"].shape[1] for x in batch]) # T_mel

    return {
        "text_ids": text_padded,
        "text_lengths": text_lengths,
        "mel_specs": mel_padded,
        "mel_lengths": mel_lengths
    }
        
# ---------------------
# 訓練主邏輯
# ---------------------
def train_tts_model(config_path: Path, data_dir: Path, output_model_path: Path, ckpt_path: Path = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(data_dir / "training.log"), logging.StreamHandler()])
    
    try:
        with open(config_path) as f:
            hps = json.load(f)
    except Exception as e:
        logging.error(f"讀取設定檔 {config_path} 失敗: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用裝置: {device}")

    try:
        dataset = TextAudioDataset(metadata_path=data_dir / "metadata.csv", hps=hps)
    except Exception as e:
        logging.error(f"初始化資料集時失敗: {e}")
        return

    if len(dataset) == 0:
        logging.error(f"資料集為空或無法載入 metadata.csv，請檢查 {data_dir / 'metadata.csv'}")
        return

    dataloader = DataLoader(
        dataset, batch_size=hps["train"]["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=hps['train']['num_workers'],
        pin_memory=True, persistent_workers=True if hps['train']['num_workers'] > 0 else False,
        drop_last=True # 避免最後一個 batch 過小導致維度問題
    )

    model_params = hps.get("model", {}) # 獲取模型超參數
    model = TextToMelModel(
        n_vocab=len(_symbols), 
        n_mels=hps['data']['n_mels'],
        hidden_dim=model_params.get("hidden_dim", 256),
        n_heads=model_params.get("n_heads", 4),
        num_encoder_layers=model_params.get("num_encoder_layers", 4),
        dim_feedforward_factor=model_params.get("dim_feedforward_factor", 4),
        dropout=model_params.get("dropout", 0.1)
    ).to(device)

    if ckpt_path and ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            logging.info(f"成功從檢查點載入模型: {ckpt_path}")
        except Exception as e:
            logging.error(f"從檢查點 {ckpt_path} 載入模型失敗: {e}")
            return


    optimizer = torch.optim.AdamW(model.parameters(), lr=hps["train"]["learning_rate"], weight_decay=hps["train"].get("weight_decay", 0.01))
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # 學習率調度器 (可選)
    scheduler = None
    if "lr_scheduler" in hps["train"]:
        scheduler_params = hps["train"]["lr_scheduler"]
        if scheduler_params.get("name") == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=hps["train"]["epochs"] * len(dataloader), 
                eta_min=scheduler_params.get("eta_min", 0)
            )
            logging.info("使用 CosineAnnealingLR 學習率調度器。")


    logging.info("訓練開始...")
    best_loss = float('inf')

    for epoch in range(hps["train"]["epochs"]):
        model.train()
        
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{hps['train']['epochs']-1}")
        
        for i, batch in enumerate(pbar):
            if batch is None: # 如果 collate_fn 返回 None (例如，整個 batch 的資料都無效)
                logging.warning(f"Epoch {epoch}, Step {i}: 收到一個空的 batch，跳過。")
                continue

            text_ids = batch["text_ids"].to(device)
            mel_specs = batch["mel_specs"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                loss = model(text_ids, mel_specs) # forward 函式現在只返回 loss

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Epoch {epoch}, Step {i}: 偵測到無效的 Loss 值 ({loss})，跳過此 batch。")
                continue

            scaler.scale(loss).backward()
            # 梯度裁剪 (可選，但有助於穩定訓練)
            scaler.unscale_(optimizer) # 在裁剪前 unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hps["train"].get("grad_clip_thresh", 1.0))
            
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            logging.info(f"Epoch {epoch} 完成 | 平均 Loss: {avg_epoch_loss:.4f} | 跳過檔案數 (本輪 dataset): {dataset.skipped_files_count}")
            dataset.skipped_files_count = 0 # 重置計數器

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                output_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_model_path)
                logging.info(f"模型已儲存 (Loss 改善): {output_model_path}")
        else:
            logging.warning(f"Epoch {epoch} 沒有處理任何有效的 batch。")


    logging.info("="*50)
    logging.info("訓練完成！")
    if dataset.skipped_files_count > 0: # 處理最後一個 epoch 可能跳過的檔案
        logging.warning(f"在最後一個 epoch 的資料準備中，跳過了 {dataset.skipped_files_count} 個檔案。")
    logging.info(f"最終模型已儲存至: {output_model_path}")
    logging.info("="*50)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="訓練 Text-to-Mel TTS 模型")
    parser.add_argument("--config", required=True, type=Path, help="設定檔路徑 (例如 configs/base_config.json)")
    parser.add_argument("--data_dir", required=True, type=Path, help="包含 metadata.csv 和音訊片段的最終訓練資料目錄")
    parser.add_argument("--output_model_path", type=Path, default=None, help="儲存訓練好的模型權重的路徑 (例如 outputs/models/my_model.pth)。如果未提供，將使用設定檔中的預設或生成一個。")
    parser.add_argument("--ckpt", type=Path, default=None, help="用於繼續訓練的檢查點模型路徑")
    args = parser.parse_args()

    output_model_p = args.output_model_path
    if output_model_p is None:
        # 如果未指定輸出路徑，則根據 data_dir 的系列名生成一個
        series_name = args.data_dir.name
        output_model_p = Path("outputs/models") / f"{series_name}_tts_model.pth"
    
    output_model_p.parent.mkdir(parents=True, exist_ok=True)


    train_tts_model(args.config, args.data_dir, output_model_p, args.ckpt)
