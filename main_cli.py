# scripts/main_cli.py (修正 tqdm 導入問題)
import subprocess
import os
from pathlib import Path
import shutil
import json
import sys
import select
import pandas as pd # <--- 新增/移至此處
from tqdm import tqdm # <--- 新增

# --- 基本設定 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

VENV_PYTHON_CANDIDATES = [
    PROJECT_ROOT / ".venv" / "bin" / "python", # Linux/macOS
    PROJECT_ROOT / ".venv" / "Scripts" / "python.exe", # Windows
]
VENV_PYTHON = None
for path_candidate in VENV_PYTHON_CANDIDATES:
    if path_candidate.exists():
        VENV_PYTHON = path_candidate
        break
if VENV_PYTHON is None:
    print(f"警告：在預期路徑中找不到虛擬環境的 Python。將嘗試使用系統預設 'python'。")
    print(f"請確保 'python' 指向您專案的虛擬環境，否則可能發生套件版本問題。")
    PYTHON_EXECUTABLE = "python"
else:
    PYTHON_EXECUTABLE = str(VENV_PYTHON)
    print(f"偵測到虛擬環境 Python: {PYTHON_EXECUTABLE}")

DATA_DIR = PROJECT_ROOT / "data"
RAW_SOURCES_BASE_DIR = DATA_DIR / "sources_raw"
DENOISED_SOURCES_BASE_DIR = DATA_DIR / "sources_denoised"
VAD_SEGMENTS_BASE_DIR = DATA_DIR / "processed" / "vad_segments"
RAW_TRANSCRIPTS_BASE_DIR = DATA_DIR / "processed" / "raw_transcripts"
TEMP_SEGMENTS_BASE_DIR = DATA_DIR / "processed" / "temp_segments_noisy"
FINAL_TRAINING_DATA_BASE_DIR = DATA_DIR / "processed" / "training_datasets"

DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "base_config.json"
DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs" / "models"
DEFAULT_SYNTHESIS_OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs" / "synthesis_results"
DEFAULT_CHAT_TEMP_AUDIO_DIR = PROJECT_ROOT / "outputs" / "temp_chat_audio"

# --- Helper Function ---
def run_command(command_parts, suppress_output=False, working_dir=None):
    """執行一個命令列指令並即時串流其輸出"""
    command_str = ' '.join(map(str, command_parts))
    effective_working_dir = working_dir if working_dir else PROJECT_ROOT
    print(f"\n▶️ 正在執行 (於 {effective_working_dir}): {command_str}")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = "1"

    process = subprocess.Popen(
        command_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', bufsize=1, universal_newlines=True,
        env=env, cwd=effective_working_dir
    )
    full_stdout, full_stderr = [], []
    if os.name != 'nt':
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            try:
                ret = select.select(reads, [], [])
                for fd in ret[0]:
                    if fd == process.stdout.fileno():
                        line = process.stdout.readline()
                        if line:
                            sys.stdout.write(line)
                            full_stdout.append(line)
                    if fd == process.stderr.fileno():
                        line = process.stderr.readline()
                        if line:
                            sys.stderr.write(line)
                            full_stderr.append(line)
            except ValueError:
                break
            if process.poll() is not None:
                break
    else:
        for stdout_line in iter(process.stdout.readline, ""):
            if stdout_line:
                sys.stdout.write(stdout_line)
                full_stdout.append(stdout_line)
        for stderr_line in iter(process.stderr.readline, ""):
            if stderr_line:
                sys.stderr.write(stderr_line)
                full_stderr.append(stderr_line)

    process.stdout.close()
    process.stderr.close()
    return_code = process.wait()

    if return_code == 0:
        if not suppress_output:
            print(f"✅ 指令成功完成 (返回碼 {return_code})")
        return True, "".join(full_stdout), "".join(full_stderr)
    else:
        print(f"❌ 指令執行失敗 (返回碼 {return_code})")
        return False, "".join(full_stdout), "".join(full_stderr)

def get_user_input(prompt, default=None, is_path=False, ensure_exists=False, is_dir=False, check_empty=True):
    while True:
        if default is not None:
            response = input(f"{prompt} (預設: {default}): ").strip()
            response = response if response else default
        else:
            response = input(f"{prompt}: ").strip()
        if check_empty and not response:
            print("輸入不能為空，請重新輸入。")
            continue
        if is_path:
            path_obj = Path(response)
            if ensure_exists:
                if not path_obj.exists():
                    print(f"錯誤：路徑 '{path_obj}' 不存在，請重新輸入。")
                    continue
                if is_dir and not path_obj.is_dir():
                    print(f"錯誤：路徑 '{path_obj}' 不是一個有效的目錄，請重新輸入。")
                    continue
            return path_obj
        return response

def ensure_directories():
    dirs_to_create = [
        RAW_SOURCES_BASE_DIR, DENOISED_SOURCES_BASE_DIR,
        VAD_SEGMENTS_BASE_DIR, RAW_TRANSCRIPTS_BASE_DIR,
        TEMP_SEGMENTS_BASE_DIR, FINAL_TRAINING_DATA_BASE_DIR,
        DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR, DEFAULT_SYNTHESIS_OUTPUT_BASE_DIR,
        DEFAULT_CHAT_TEMP_AUDIO_DIR, PROJECT_ROOT / "configs"
    ]
    for d_path in dirs_to_create:
        d_path.mkdir(parents=True, exist_ok=True)
    print("預設目錄結構已檢查/建立。")


# --- 各個流程的函式 ---
def step_download_audio():
    print("\n--- 步驟 [dl]: 下載 YouTube 音訊 ---")
    yt_url = get_user_input("請輸入 YouTube URL (影片或播放列表)", check_empty=True)
    series_name = get_user_input("請為此音訊系列命名 (例如 MySinger_AlbumX)", check_empty=True)
    cmd = [
        PYTHON_EXECUTABLE, str(SCRIPTS_DIR / "download_audio.py"),
        yt_url, "--series_name", series_name, "--output_base_dir", str(RAW_SOURCES_BASE_DIR)
    ]
    success, _, _ = run_command(cmd)
    return RAW_SOURCES_BASE_DIR / series_name if success else None

def step_denoise_long_audio(input_series_dir_name: str = None):
    print("\n--- 步驟 [dnl]: (可選) 對原始長音訊進行降噪 ---")
    if not input_series_dir_name:
        input_series_dir_name = get_user_input(f"請輸入位於 '{RAW_SOURCES_BASE_DIR}' 下的原始音訊系列目錄名", check_empty=True)
    raw_audio_dir = RAW_SOURCES_BASE_DIR / input_series_dir_name
    if not raw_audio_dir.exists() or not raw_audio_dir.is_dir():
        print(f"錯誤：找不到原始音訊目錄 {raw_audio_dir}")
        return None
    denoised_output_dir = DENOISED_SOURCES_BASE_DIR / input_series_dir_name
    cmd = [
        PYTHON_EXECUTABLE, str(SCRIPTS_DIR / "denoise_audio_final.py"), # 假設您的降噪腳本名為 denoise_audio_final.py
        "--input_dir", str(raw_audio_dir), "--output_dir", str(denoised_output_dir)
    ]
    success, _, _ = run_command(cmd)
    return denoised_output_dir if success else None

def step_vad_segmentation(input_series_dir_name: str = None, use_denoised: bool = True):
    print("\n--- 步驟 [vad]: VAD 智能切割 (移除靜音) ---")
    base_dir_to_use = DENOISED_SOURCES_BASE_DIR if use_denoised else RAW_SOURCES_BASE_DIR
    prompt_msg = f"請輸入位於 '{base_dir_to_use}' 下的{'已降噪' if use_denoised else '原始'}音訊系列目錄名進行 VAD 切割"

    if not input_series_dir_name:
        input_series_dir_name = get_user_input(prompt_msg, check_empty=True)

    audio_dir_to_segment = base_dir_to_use / input_series_dir_name
    if not audio_dir_to_segment.exists() or not audio_dir_to_segment.is_dir():
        print(f"錯誤：找不到音訊目錄 {audio_dir_to_segment} 進行 VAD 切割。")
        return None

    vad_output_dir = VAD_SEGMENTS_BASE_DIR / input_series_dir_name

    cmd = [
        PYTHON_EXECUTABLE, str(SCRIPTS_DIR / "vad_segmenter.py"),
        "--input_dir", str(audio_dir_to_segment),
        "--output_dir", str(vad_output_dir)
    ]
    success, _, _ = run_command(cmd)
    return vad_output_dir if success else None

def step_transcribe_audio(input_series_dir_name: str = None):
    print("\n--- 步驟 [tr]: 使用 Whisper 轉錄 (VAD切割後的片段) ---")
    base_dir_to_use = VAD_SEGMENTS_BASE_DIR
    prompt_msg = f"請輸入位於 '{base_dir_to_use}' 下的 VAD 已切割系列目錄名進行轉錄"

    if not input_series_dir_name:
        input_series_dir_name = get_user_input(prompt_msg, check_empty=True)

    audio_dir_to_transcribe = base_dir_to_use / input_series_dir_name
    if not audio_dir_to_transcribe.exists() or not audio_dir_to_transcribe.is_dir():
        print(f"錯誤：找不到 VAD 已切割的音訊目錄 {audio_dir_to_transcribe} 進行轉錄。")
        return None

    transcript_output_dir = RAW_TRANSCRIPTS_BASE_DIR / input_series_dir_name

    language = get_user_input("請輸入音訊語言 (例如 ja, en, zh)", "ja")
    model_size = get_user_input("請輸入 Whisper 模型大小 (例如 medium, large-v3)", "medium")

    cmd = [
        PYTHON_EXECUTABLE, str(SCRIPTS_DIR / "transcribe_audio.py"),
        "--input_dir", str(audio_dir_to_transcribe),
        "--output_dir", str(transcript_output_dir),
        "--language", language,
        "--model_size", model_size
    ]
    success, _, _ = run_command(cmd)
    return transcript_output_dir if success else None

def step_segment_audio_ffmpeg(transcript_metadata_path: Path = None,
                              audio_source_for_segmentation_dir_name: str = None,
                              use_denoised_for_segmentation: bool = True):
    print("\n--- 步驟 [seg]: (舊流程) 使用 FFmpeg 根據時間戳分割音訊 ---")
    print("警告：此步驟在新 VAD 流程中已被 [vad] 步驟取代，通常無需執行。")
    if not transcript_metadata_path:
        series_name_for_meta = get_user_input(f"請輸入位於 '{RAW_TRANSCRIPTS_BASE_DIR}' 下的轉錄系列目錄名以獲取 metadata.csv", check_empty=True)
        transcript_metadata_path = RAW_TRANSCRIPTS_BASE_DIR / series_name_for_meta / "metadata.csv"

    if not transcript_metadata_path.exists():
        print(f"錯誤：找不到轉錄 metadata 檔案 {transcript_metadata_path}")
        return None

    audio_base_dir = DENOISED_SOURCES_BASE_DIR if use_denoised_for_segmentation else RAW_SOURCES_BASE_DIR
    if not audio_source_for_segmentation_dir_name:
        audio_source_for_segmentation_dir_name = get_user_input(
            f"請輸入位於 '{audio_base_dir}' 下的{'已降噪' if use_denoised_for_segmentation else '原始'}長音訊系列目錄名 (用於分割)",
            default=transcript_metadata_path.parent.name
        )

    long_audio_parent_dir = audio_base_dir / audio_source_for_segmentation_dir_name
    if not long_audio_parent_dir.exists() or not long_audio_parent_dir.is_dir():
        print(f"錯誤：找不到用於分割的長音訊目錄 {long_audio_parent_dir}")
        return None

    temp_segments_output_dir = TEMP_SEGMENTS_BASE_DIR / audio_source_for_segmentation_dir_name

    segment_script_path = SCRIPTS_DIR / "segment_audio.py" # 假設舊的分割腳本名
    if not segment_script_path.exists():
        print(f"錯誤：找不到舊的分割腳本 {segment_script_path}。無法執行此步驟。")
        return None

    cmd = [
        PYTHON_EXECUTABLE, str(segment_script_path),
        "--metadata_path", str(transcript_metadata_path),
        "--raw_audio_parent_dir", str(long_audio_parent_dir),
        "--output_dir", str(temp_segments_output_dir)
    ]
    success, _, _ = run_command(cmd)
    return temp_segments_output_dir if success else None

def step_filter_denoise_segments(transcribed_series_dir_name: str = None, min_duration: float = 0.5):
    print("\n--- 步驟 [flt]: 過濾/整理片段 (並可選二次降噪) ---")
    if not transcribed_series_dir_name:
        transcribed_series_dir_name = get_user_input(f"請輸入位於 '{RAW_TRANSCRIPTS_BASE_DIR}' 下的已轉錄系列目錄名", check_empty=True)

    input_transcript_dir = RAW_TRANSCRIPTS_BASE_DIR / transcribed_series_dir_name
    input_audio_dir = VAD_SEGMENTS_BASE_DIR / transcribed_series_dir_name # 音訊現在來自 VAD 目錄

    if not input_transcript_dir.exists() or not (input_transcript_dir / "metadata.csv").exists():
        print(f"錯誤：找不到轉錄目錄 {input_transcript_dir} 或其下的 metadata.csv")
        return None
    if not input_audio_dir.exists():
        print(f"錯誤：找不到對應的 VAD 音訊目錄 {input_audio_dir}")
        return None

    # denoise_audio_final.py 的短片段模式需要音訊和 metadata.csv 在同一個輸入目錄
    # 我們建立一個臨時整合目錄，並將 VAD 的音訊與 Whisper 的 metadata 複製進去
    temp_filter_input_dir = TEMP_SEGMENTS_BASE_DIR / f"{transcribed_series_dir_name}_for_filtering"
    temp_filter_input_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在準備過濾：將音訊與 metadata 複製到臨時目錄 {temp_filter_input_dir}...")

    source_meta_path = input_transcript_dir / "metadata.csv"
    target_meta_path = temp_filter_input_dir / "metadata.csv"

    # 讀取 transcribe_audio.py 產生的 metadata
    df = pd.read_csv(source_meta_path)
    # denoise_audio_final.py 的短片段模式期望的檔名欄位是 'segment_filename'
    # transcribe_audio.py 產生的檔名欄位是 'filename'
    if 'filename' in df.columns:
        df.rename(columns={'filename': 'segment_filename'}, inplace=True)
    else:
        print(f"錯誤: {source_meta_path} 中缺少 'filename' 欄位。")
        return None
    # 確保 'text' 欄位存在，denoise_audio_final.py 也會用到
    if 'text' not in df.columns:
        print(f"警告: {source_meta_path} 中缺少 'text' 欄位，將用空字串填充。")
        df['text'] = ""

    df.to_csv(target_meta_path, index=False, encoding='utf-8')
    print(f"Metadata 已複製並調整欄位名至 {target_meta_path}")

    # 複製 VAD 切割後的音訊檔案到臨時目錄
    # 使用 metadata 中的檔名列表來確定要複製哪些檔案
    files_to_copy_from_metadata = df['segment_filename'].unique()
    copied_count = 0
    for audio_filename in tqdm(files_to_copy_from_metadata, desc="複製音訊檔案到臨時目錄"):
        source_audio_file = input_audio_dir / audio_filename
        if source_audio_file.exists():
            shutil.copy(source_audio_file, temp_filter_input_dir / audio_filename)
            copied_count += 1
        else:
            print(f"警告：在 {source_meta_path} 中列出的檔案 {audio_filename} 在 VAD 音訊目錄 {input_audio_dir} 中未找到，已跳過。")
    print(f"已複製 {copied_count} 個音訊檔案到 {temp_filter_input_dir}")


    final_output_dir = FINAL_TRAINING_DATA_BASE_DIR / transcribed_series_dir_name

    cmd = [
        PYTHON_EXECUTABLE, str(SCRIPTS_DIR / "denoise_audio.py"), # 假設您的降噪腳本名為 denoise_audio_final.py
        "--input_dir", str(temp_filter_input_dir),
        "--output_dir", str(final_output_dir),
        "--min_duration", str(min_duration)
        # batch_size 等其他參數會使用 denoise_audio_final.py 中的預設值
    ]
    success, _, _ = run_command(cmd)

    print(f"臨時目錄 {temp_filter_input_dir} 已保留，方便除錯。可手動刪除。")
    return final_output_dir if success else None

def step_train_tts_model(training_data_dir_name: str = None, config_file_path: Path = None, output_model_name_prefix: str = None):
    print("\n--- 步驟 [train]: 訓練 TTS 模型 ---")
    if not training_data_dir_name:
        training_data_dir_name = get_user_input(f"請輸入位於 '{FINAL_TRAINING_DATA_BASE_DIR}' 下的最終訓練資料系列目錄名", check_empty=True)

    final_training_dir = FINAL_TRAINING_DATA_BASE_DIR / training_data_dir_name
    if not final_training_dir.exists() or not final_training_dir.is_dir() or not (final_training_dir / "metadata.csv").exists():
        print(f"錯誤：找不到最終訓練資料目錄 {final_training_dir} 或其下的 metadata.csv")
        return False

    if not config_file_path:
        config_file_path = get_user_input("請輸入模型設定檔路徑", default=str(DEFAULT_TRAIN_CONFIG_PATH), is_path=True, ensure_exists=True)
    if not config_file_path: return False

    if not output_model_name_prefix:
        output_model_name_prefix = get_user_input("請為輸出的模型檔案設定一個前綴 (例如 MyModelV1)", default=training_data_dir_name)

    output_model_path = DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR / f"{output_model_name_prefix}_tts_model.pth"

    # 【修改】調整 ckpt_path 的獲取方式，允許直接按 Enter 表示 None
    ckpt_prompt = "若要繼續訓練，請輸入檢查點模型路徑 (.pth) (直接按 Enter 表示不使用): "
    ckpt_path_str = input(ckpt_prompt).strip() # 直接使用 input，並去除前後空格
    
    ckpt_path = None
    if ckpt_path_str: # 只有當使用者確實輸入了內容時才嘗試轉換為 Path
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            print(f"警告：提供的檢查點路徑 '{ckpt_path_str}' 不存在，將從頭開始訓練。")
            ckpt_path = None # 路徑無效，重置為 None
    else:
        print("不使用檢查點，將從頭開始訓練或根據已有模型（如果 output_model_path 已存在）繼續。")

    train_script_module = "scripts.train_tts" # 假設您的訓練腳本可以作為模組執行
    # 檢查訓練腳本是否存在，如果您的訓練腳本不是以 -m 方式執行的，請相應修改
    # if not (SCRIPTS_DIR / "train_tts.py").exists():
    #     print(f"錯誤：找不到訓練腳本 {SCRIPTS_DIR / 'train_tts.py'}")
    #     return False

    cmd = [
        PYTHON_EXECUTABLE, "-m", train_script_module,
        "--config", str(config_file_path),
        "--data_dir", str(final_training_dir),
        "--output_model_path", str(output_model_path)
    ]
    if ckpt_path:
        cmd.extend(["--ckpt", str(ckpt_path)])

    success, _, _ = run_command(cmd)
    if success:
        print(f"🎉 訓練完成！模型已儲存至 (或更新於): {output_model_path}")
    return success

def step_synthesize_speech(model_path_str: str = None):
    print("\n--- 步驟 [synth]: 合成語音 ---")
    if not model_path_str:
        model_path_str = get_user_input(f"請輸入 TTS 模型檔案路徑 (.pth) 或位於 '{DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR}' 下的模型前綴名", check_empty=True)

    tts_model_path = Path(model_path_str)
    if not tts_model_path.is_file():
        potential_path = DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR / f"{model_path_str}_tts_model.pth"
        if potential_path.is_file():
            tts_model_path = potential_path
        else:
            potential_path_no_suffix = DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR / model_path_str
            if potential_path_no_suffix.is_file() and potential_path_no_suffix.suffix == ".pth":
                tts_model_path = potential_path_no_suffix
            else:
                print(f"錯誤：找不到 TTS 模型檔案。嘗試路徑: {model_path_str} 和 {potential_path}")
                return

    print(f"使用 TTS 模型: {tts_model_path}")
    output_dir = DEFAULT_SYNTHESIS_OUTPUT_BASE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    input_type = get_user_input("輸入類型 (1: 單一句子, 2: 文字檔案): ", "1")

    synth_script_module = "scripts.synthesize" # 假設您的合成腳本可以作為模組執行
    # if not (SCRIPTS_DIR / "synthesize.py").exists():
    #     print(f"錯誤：找不到合成腳本 {SCRIPTS_DIR / 'synthesize.py'}")
    #     return

    cmd_synth_base = [
        PYTHON_EXECUTABLE, "-m", synth_script_module,
        "--model_path", str(tts_model_path)
    ]
    if input_type == "1":
        text_to_synth = get_user_input("請輸入要合成的日文句子", check_empty=True)
        output_filename = get_user_input("請輸入輸出 .wav 檔名 (不含路徑)", default=f"synth_{Path(text_to_synth[:10].replace(' ','_')).stem}.wav")
        output_wav_path = output_dir / output_filename
        cmd_synth_base.extend(["--text", text_to_synth, "--output_path", str(output_wav_path)])
    elif input_type == "2":
        text_file_path = get_user_input("請輸入文字檔案路徑 (.txt)", is_path=True, ensure_exists=True, check_empty=True)
        output_filename = get_user_input("請輸入合併輸出的 .wav 檔名 (不含路徑)", default=f"synth_batch_{text_file_path.stem}.wav")
        output_wav_path = output_dir / output_filename
        cmd_synth_base.extend(["--text_file", str(text_file_path), "--output_path", str(output_wav_path)])
    else:
        print("無效的輸入類型。")
        return
    success, _, _ = run_command(cmd_synth_base)
    if success:
        print(f"🎉 合成結束！檔案位於 {output_wav_path}")

def step_interactive_chat(model_path_str: str = None):
    print("\n--- 步驟 [chat]: 互動式聊天 ---")
    if not model_path_str:
        model_path_str = get_user_input(f"請輸入 TTS 模型檔案路徑 (.pth) 或位於 '{DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR}' 下的模型前綴名", check_empty=True)
    tts_model_path = Path(model_path_str)
    if not tts_model_path.is_file():
        potential_path = DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR / f"{model_path_str}_tts_model.pth"
        if potential_path.is_file():
            tts_model_path = potential_path
        else:
            potential_path_no_suffix = DEFAULT_TTS_MODEL_OUTPUT_BASE_DIR / model_path_str
            if potential_path_no_suffix.is_file() and potential_path_no_suffix.suffix == ".pth":
                tts_model_path = potential_path_no_suffix
            else:
                print(f"錯誤：找不到 TTS 模型檔案。嘗試路徑: {model_path_str} 和 {potential_path}")
                return
    print(f"使用 TTS 模型: {tts_model_path}")
    llm_model_name = get_user_input("請輸入 Ollama LLM 模型名稱 (例如 qwen:7b)", default="qwen:7b")

    chat_script_module = "scripts.chat" # 假設您的聊天腳本可以作為模組執行
    # if not (SCRIPTS_DIR / "chat.py").exists():
    #     print(f"錯誤：找不到聊天腳本 {SCRIPTS_DIR / 'chat.py'}")
    #     return

    cmd_chat = [
        PYTHON_EXECUTABLE, "-m", chat_script_module,
        "--model_path", str(tts_model_path),
        "--llm_model", llm_model_name
    ]
    print("正在啟動互動式聊天...")
    print(f"您可以手動執行以下指令（如果自動啟動失敗）：\n{' '.join(cmd_chat)}")
    try:
        run_command(cmd_chat)
    except KeyboardInterrupt:
        print("\n互動式聊天被中斷。")
    except Exception as e:
        print(f"啟動互動式聊天時發生錯誤: {e}")


def full_pipeline_execution():
    print("\n===== 開始【新版 VAD】完整資料準備與訓練流程 =====")
    print("流程：下載 -> 降噪長音訊 -> VAD智能切割 -> Whisper轉錄 -> 過濾/整理短片段 -> 訓練")
    print("-" * 30)

    # 1. 下載
    raw_audio_series_output_dir = step_download_audio()
    if not raw_audio_series_output_dir: return
    current_series_name = raw_audio_series_output_dir.name

    # 2. 降噪長音訊
    denoised_long_audio_output_dir = step_denoise_long_audio(current_series_name)
    if not denoised_long_audio_output_dir: return

    # 3. VAD 智能切割 (使用降噪後的音訊)
    vad_segments_output_dir = step_vad_segmentation(current_series_name, use_denoised=True)
    if not vad_segments_output_dir: return

    # 4. Whisper 轉錄 (對 VAD 切割後的片段)
    transcribed_output_dir = step_transcribe_audio(current_series_name)
    if not transcribed_output_dir: return

    # 5. 過濾/整理短片段
    min_duration_input = get_user_input("請輸入過濾短片段的最小時長 (秒)", "0.5")
    try:
        min_duration = float(min_duration_input)
    except ValueError:
        print(f"錯誤的時長輸入 '{min_duration_input}'，將使用預設值 0.5 秒。")
        min_duration = 0.5
        
    final_training_data_output_dir = step_filter_denoise_segments(
        current_series_name,
        min_duration
    )
    if not final_training_data_output_dir: return

    # 6. 訓練
    step_train_tts_model(current_series_name, DEFAULT_TRAIN_CONFIG_PATH, current_series_name)
    print("===== 完整 VAD 流程執行完畢 =====")

def main():
    ensure_directories()
    global PYTHON_EXECUTABLE
    if VENV_PYTHON is None:
        # This was already printed at the top if VENV_PYTHON is None
        # print(f"警告：在預期路徑中找不到虛擬環境的 Python。將嘗試使用系統預設 'python'。")
        # print(f"請確保 'python' 指向您專案的虛擬環境，否則可能發生套件版本問題。")
        PYTHON_EXECUTABLE = "python"
    else:
        PYTHON_EXECUTABLE = str(VENV_PYTHON)
        # print(f"將使用虛擬環境 Python: {PYTHON_EXECUTABLE}") # This was already printed at the top

    while True:
        print("\n\n======== TTS 日語語音專案 CLI 主控台 (VAD 增強版) ========")
        print("推薦流程:")
        print("[1] 【推薦】完整 VAD 流程：下載 -> 降噪 -> VAD切割 -> 轉錄 -> 過濾 -> 訓練")
        print("-" * 40)
        print("單獨步驟:")
        print("  [dl]  下載 YouTube 音訊")
        print("  [dnl] 降噪長音訊 (需先 [dl])")
        print("  [vad] VAD 智能切割 (需先 [dl] 或 [dnl])")
        print("  [tr]  Whisper 轉錄 (需先 [vad])")
        print("  [seg] (舊流程) FFmpeg 分割 (已被 [vad] 取代)")
        print("  [flt] 過濾/整理分割片段 (需先 [tr])")
        print("  [train] 訓練 TTS 模型 (需先 [flt])")
        print("-" * 40)
        print("應用功能:")
        print("  [synth] 合成語音")
        print("  [chat]  互動式聊天")
        print("-" * 40)
        print("[0] 退出")

        choice = input("請輸入選項: ").lower().strip()

        if choice == '1':
            full_pipeline_execution()
        elif choice == 'dl':
            step_download_audio()
        elif choice == 'dnl':
            step_denoise_long_audio()
        elif choice == 'vad':
            use_denoised_q = get_user_input("是否對已降噪的音訊進行 VAD 切割? (y/n)", "y").lower()
            step_vad_segmentation(use_denoised=(use_denoised_q == 'y'))
        elif choice == 'tr':
            step_transcribe_audio()
        elif choice == 'seg':
            print("警告：此為舊版 FFmpeg 分割流程，在新版 VAD 流程中通常不需要。")
            use_denoised_q = get_user_input("用於分割的長音訊是否是已降噪版本? (y/n)", "y").lower()
            step_segment_audio_ffmpeg(use_denoised_for_segmentation=(use_denoised_q == 'y'))
        elif choice == 'flt':
            step_filter_denoise_segments()
        elif choice == 'train':
            step_train_tts_model()
        elif choice == 'synth':
            step_synthesize_speech()
        elif choice == 'chat':
            step_interactive_chat()
        elif choice == '0':
            print("感謝使用，再見！")
            break
        else:
            print("無效的選項，請重新輸入。")

if __name__ == "__main__":
    main()