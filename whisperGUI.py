import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from faster_whisper import WhisperModel
import threading
import os
import sys
from queue import Queue
import logging
from typing import Optional
from datetime import datetime
import requests
import inspect
from pathlib import Path
import subprocess
import ctypes
import winreg
import json
import torch  # 添加缺失的导入

def monitor_memory(self):
    """显存使用监控"""
    if self._device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logger.debug(f"显存使用: 已分配 {allocated:.2f}GB / 缓存 {cached:.2f}GB")
        if allocated > 0.9 * torch.cuda.max_memory_allocated():
            logger.warning("显存即将耗尽，建议降低批大小")

def check_env():
    """环境健康检查"""
    issues = []
    if not torch.cuda.is_available():
        issues.append("未检测到NVIDIA显卡驱动")
    if torch.cuda.device_count() == 0:
        issues.append("没有可用的CUDA设备")
    return issues

def get_app_dir() -> Path:
    """获取应用程序运行目录
    
    根据运行环境(打包或开发)返回正确的程序目录
    
    Returns:
        Path: 应用程序根目录路径
    """
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent

def format_srt_time(seconds: float) -> str:
    """将秒数格式化为SRT字幕时间格式
    
    Args:
        seconds: 秒数，可以是浮点数
        
    Returns:
        str: 格式为HH:MM:SS,mmm的SRT时间字符串
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

# 配置国内镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 日志配置
def setup_logger():
    logger = logging.getLogger("whisper_transcriber")
    logger.setLevel(logging.INFO)

    log_dir = get_app_dir() / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"transcribe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 初始化日志
logger = setup_logger()

# 模型路径
def get_model_dir():
    return get_app_dir() / "models"

# 检查本地模型文件夹
def check_local_base_model():
    model_dir = get_model_dir()
    if model_dir.exists() and any(model_dir.iterdir()):
        logger.info("找到本地模型文件夹")
        return str(model_dir)
    return None

# 检查FFmpeg安装
def check_ffmpeg_installed():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except (FileNotFoundError, PermissionError):
        return False

class WhisperTranscriber:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcriber")
        self.root.geometry("800x650")
        self.root.configure(bg="#f5f5f5")
        self.root.resizable(True, True)

        self.font = ("SimHei", 10)
        self.message_queue = Queue()
        self.root.after(100, self.process_messages)

        self.using_cpu = True
        self.local_base_model = check_local_base_model()

        self.config_file = get_app_dir() / "config.json"
        self.config = self.load_config()

        self.create_ui()
        self.model_cache = {}
        self.check_ffmpeg()

        logger.info(f"应用已启动 - 运行目录: {get_app_dir()}")

        self.result_frame_initialized = False
        self.requests_session = None

    def load_config(self):
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error("配置文件解析失败，使用默认配置")
        return {}

    def save_config(self):
        """保存配置到JSON文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False

    def create_ui(self):
        main_frame = tk.Frame(self.root, bg="#f5f5f5")
        main_frame.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=1)
        for i in range(7):
            main_frame.rowconfigure(i, weight=0)
        main_frame.rowconfigure(6, weight=1)

        # 顶部信息
        top_frame = tk.Frame(main_frame, bg="#f5f5f5")
        top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(
            top_frame, text="本应用用于音频/视频转录", bg="#f5f5f5", fg="#555555", font=(self.font[0], 10, "italic")
        ).pack(side=tk.LEFT, padx=5)

        # 模型配置区域
        config_frame = tk.LabelFrame(main_frame, text="模型配置", bg="#f5f5f5", padx=10, pady=10)
        config_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        for i in range(6):
            config_frame.columnconfigure(i, weight=1)

        # 模型选择
        tk.Label(config_frame, text="模型大小", bg="#f5f5f5", font=self.font).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_size = ttk.Combobox(config_frame, values=["tiny", "base", "small", "medium", "large-v2"], width=12)
        self.model_size.set("base")
        self.model_size.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # 设备选择
        tk.Label(config_frame, text="设备", bg="#f5f5f5", font=self.font).grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self.accelerate_btn = tk.Button(
            config_frame, text="加速", command=self.toggle_acceleration,
            bg="#FF9800", fg="white", padx=5, font=self.font
        )
        self.accelerate_btn.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # 语言选择
        tk.Label(config_frame, text="语言", bg="#f5f5f5", font=self.font).grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.language = ttk.Combobox(config_frame, values=["auto"], width=10)
        self.language.set("auto")
        self.language.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        # 模型路径
        tk.Label(config_frame, text="模型路径", bg="#f5f5f5", font=self.font).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_path = tk.Entry(config_frame, width=30, font=self.font)
        self.model_path.grid(row=1, column=1, columnspan=4, padx=5, pady=5, sticky="ew")

        # 如果有本地模型文件夹，自动填入路径
        if self.local_base_model:
            self.model_path.delete(0, tk.END)
            self.model_path.insert(0, self.local_base_model)
            self.model_size.config(state=tk.DISABLED)
            logger.info(f"已检测到本地base模型，路径已自动填充: {self.local_base_model}")

        # 从配置文件加载模型路径
        if 'model.path' in self.config:
            self.model_path.delete(0, tk.END)
            self.model_path.insert(0, self.config['model.path'])
            self.model_size.config(state=tk.DISABLED)
            logger.info(f"从配置文件加载模型路径: {self.config['model.path']}")

        # 从配置文件加载设备设置
        if 'model.device' in self.config and self.config['model.device'] == 'cuda':
            self.using_cpu = False
            self.accelerate_btn.config(text="GPU模式", bg="#4CAF50")

        self.browse_btn = tk.Button(
            config_frame, text="浏览", command=self.select_model_dir,
            bg="#4CAF50", fg="white", padx=5, font=self.font
        )
        self.browse_btn.grid(row=1, column=5, padx=5, pady=5, sticky="w")

        # 文件选择区域
        file_frame = tk.LabelFrame(main_frame, text="音频文件", bg="#f5f5f5", padx=10, pady=10)
        file_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        file_frame.columnconfigure(1, weight=1)

        self.file_btn = tk.Button(
            file_frame, text="选择文件", command=self.select_file, bg="#4CAF50", fg="white", padx=10, font=self.font
        )
        self.file_btn.grid(row=0, column=0, padx=5, pady=5)

        self.file_label = tk.Label(
            file_frame, text="未选择文件", bg="#e0e0e0", anchor="w", padx=10, relief="sunken", font=self.font
        )
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # 输出格式选择
        output_frame = tk.LabelFrame(main_frame, text="输出格式", bg="#f5f5f5", padx=10, pady=10)
        output_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.txt_var = tk.BooleanVar(value=True)
        self.srt_var = tk.BooleanVar(value=True)

        tk.Checkbutton(
            output_frame, text="TXT格式", variable=self.txt_var, bg="#f5f5f5", font=self.font
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        tk.Checkbutton(
            output_frame, text="SRT格式", variable=self.srt_var, bg="#f5f5f5", font=self.font
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # 转录按钮
        self.transcribe_btn = tk.Button(
            main_frame, text="开始转录", command=self.start_transcription, state=tk.DISABLED,
            bg="#2196F3", fg="white", font=(self.font[0], 11, "bold"), padx=20, pady=10
        )
        self.transcribe_btn.grid(row=4, column=0, pady=15)

        # 进度条
        self.progress_frame = tk.Frame(main_frame, bg="#f5f5f5")
        self.progress_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        self.progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            self.progress_frame, orient="horizontal", variable=self.progress_var,
            length=100, mode="determinate"
        )
        self.progress.pack(fill=tk.X, expand=True, padx=5, pady=5)

        self.progress_label = tk.Label(self.progress_frame, text="准备就绪", bg="#f5f5f5", font=self.font)
        self.progress_label.pack(pady=2)

        # 结果框
        self.result_frame = tk.LabelFrame(main_frame, text="转录结果", bg="#f5f5f5", padx=10, pady=10)
        self.result_frame.grid(row=6, column=0, sticky="nsew", padx=5, pady=5)
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)

        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w",
            bg="#e0e0e0", fg="#333333", font=self.font
        )
        status_bar.pack(side="bottom", fill="x")

    def select_file(self):
        """选择音频/视频文件"""
        filetypes = (
            ('音频文件', '*.mp3 *.wav *.m4a *.flac *.ogg'),
            ('视频文件', '*.mp4 *.avi *.mov *.mkv'),
            ('所有文件', '*.*')
        )

        filename = filedialog.askopenfilename(
            title="选择文件", filetypes=filetypes
        )

        if filename:
            self.filename = filename
            self.file_label.config(text=os.path.basename(filename))
            self.transcribe_btn.config(state=tk.NORMAL)
            self.status_var.set(f"已选择文件: {os.path.basename(filename)}")
            logger.info(f"已选择文件: {filename}")

    def select_model_dir(self):
        """选择本地模型目录"""
        dirname = filedialog.askdirectory(title="浏览")
        #优先在本地模型目录中查找相对路径{}\models
        if dirname.endswith("models"):
            dirname = os.path.join(dirname, "base")
        if dirname and os.path.exists(dirname):
            self.model_path.delete(0, tk.END)
            self.model_path.insert(0, dirname)
            self.model_size.config(state=tk.DISABLED)
            logger.info(f"已选择本地模型路径: {dirname}")

            # 保存模型路径到配置
            self.config["model.path"] = dirname
            if not self.save_config():
                logger.error("保存模型路径到配置文件失败")
                messagebox.showerror(
                    "错误",
                    "保存模型路径到配置文件失败"
                )
        else:
            self.model_size.config(state=tk.NORMAL)
            logger.info("取消选择本地模型路径")

    def toggle_acceleration(self):
        """切换CPU/GPU模式"""
        self.using_cpu = not self.using_cpu
        if self.using_cpu:
            self.accelerate_btn.config(text="加速", bg="#FF9800")
            logger.info("切换到CPU模式")
            self.config["model.device"] = "cpu"
        else:
            self.accelerate_btn.config(text="GPU模式", bg="#4CAF50")
            logger.info("切换到GPU模式")
            self.config["model.device"] = "cuda"

        if not self.save_config():
            logger.error("保存设备设置到配置文件失败")
            messagebox.showerror(
                "错误",
                "保存设备设置到配置文件失败"
            )

    def check_ffmpeg(self):
        """检查并安装FFmpeg"""
        if not check_ffmpeg_installed():
            logger.info("未检测到FFmpeg")
            ffmpeg_path = get_app_dir() / "ffmpeg"

            if ffmpeg_path.exists() and ffmpeg_path.is_dir():
                logger.info(f"找到FFmpeg安装包: {ffmpeg_path}")

                # 询问用户是否安装
                answer = messagebox.askyesno(
                    "依赖检查",
                    "是否安装FFmpeg?\n\n安装包路径: " + str(ffmpeg_path)
                )

                if answer:
                    threading.Thread(target=self.install_ffmpeg, daemon=True).start()
                else:
                    messagebox.showwarning(
                        "依赖缺失",
                        "某些音频格式可能无法正常处理。\n请手动安装FFmpeg并确保已添加到系统PATH。"
                    )
            else:
                logger.warning(f"未找到FFmpeg安装包: {ffmpeg_path}")
                messagebox.showwarning(
                    "依赖缺失",
                    "未检测到FFmpeg且未找到安装包。\n"
                    "请确保安装包位于程序目录下的ffmpeg文件夹中。\n"
                    "某些音频格式可能无法正常处理。"
                )
        else:
            logger.info("FFmpeg已安装")

    def install_ffmpeg(self):
        """安装FFmpeg并添加到环境变量"""
        self.send_message("status", "正在安装FFmpeg")
        try:
            ffmpeg_path = get_app_dir() / "ffmpeg"

            # 检查bin目录
            ffmpeg_bin = ffmpeg_path / "bin"
            if not (ffmpeg_bin.exists() and ffmpeg_bin.is_dir()):
                raise FileNotFoundError(f"FFmpeg bin目录不存在: {ffmpeg_bin}")

            # 获取当前PATH
            path_var = os.environ['PATH']

            # 检查是否已在PATH中
            if str(ffmpeg_bin) not in path_var:
                # 更新PATH环境变量
                new_path = f"{str(ffmpeg_bin)}{os.pathsep}{path_var}"

                # 写入注册表（针对当前用户）
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    "Environment",
                    0,
                    winreg.KEY_SET_VALUE
                )
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                winreg.CloseKey(key)

                # 通知系统环境变量已更改
                ctypes.windll.user32.SendMessageW(
                    0xFFFF, 0x001A, 0, 0
                )

                logger.info("已将FFmpeg添加到系统PATH环境变量")
            else:
                logger.info("FFmpeg路径已在系统PATH中")

            # 安装成功提示
            self.send_message("message", ("info", "安装完成", "FFmpeg安装成功\n\n请重启应用"))
            self.send_message("status", "准备就绪")
            logger.info("FFmpeg安装成功")

        except Exception as e:
            self.send_message("message", ("error", "安装失败", f"FFmpeg安装失败\n错误详情: {str(e)}"))
            self.send_message("status", "准备就绪")
            logger.exception("FFmpeg安装失败")

    def start_transcription(self):
        """开始转录（前置检查）"""
        if not hasattr(self, 'filename'):
            messagebox.showwarning("提示", "请选择文件")
            logger.warning("用户尝试在未选择文件的情况下开始转录")
            return

        # 检查是否至少选择了一种输出格式
        if not self.txt_var.get() and not self.srt_var.get():
            messagebox.showwarning("提示", "请选择至少一种输出格式")
            logger.warning("用户未选择任何输出格式")
            return

        # 记录转录开始
        logger.info(f"开始转录 - 文件: {os.path.basename(self.filename)}, 模型: {self.model_size.get()}, 设备: {'CPU' if self.using_cpu else 'GPU'}")

        # 禁用UI元素防止重复操作
        self.transcribe_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
        self.model_size.config(state=tk.DISABLED)
        self.accelerate_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        self.language.config(state=tk.DISABLED)

        # 初始化结果框
        self.create_result_frame()

        # 重置进度条
        self.progress_var.set(0)
        self.progress_label.config(text="正在准备...")

        # 在新线程中运行转录（避免UI卡顿）
        threading.Thread(target=self.run_transcription, daemon=True).start()

    def run_transcription(self):
        """执行转录逻辑（线程中运行）"""
        try:
            model_size = self.model_size.get()
            device = "cpu" if self.using_cpu else "cuda"
            lang = self.language.get()
            model_path = self.model_path.get().strip()

            # 自动检测时语言设为None
            language = lang if lang != "auto" else None

            # 更新状态
            if model_path:
                self.send_message("status", "正在加载模型")
                logger.info(f"正在从 {os.path.basename(model_path)} 加载模型...")
            else:
                model_dir = get_model_dir()
                self.send_message("status", f"正在下载 {model_size} 模型（国内镜像加速）...")
                logger.info(f"正在下载模型: {model_size} 到 {model_dir}")
            self.send_message("progress", 10)

            # 检查模型缓存（避免重复加载）
            model_key = f"{model_size}_{device}_{model_path}"
            if model_key not in self.model_cache:
                # 加载模型
                if model_path:
                    try:
                        self.model_cache[model_key] = WhisperModel(
                            model_path, device=device, compute_type="int8"
                        )
                        logger.info(f"已从本地加载模型: {model_path}")
                    except RuntimeError as e:
                        if "CUDA driver version is insufficient" in str(e):
                            # 处理CUDA驱动版本不足的情况
                            logger.warning("CUDA驱动版本不足，尝试回退到CPU模式")
                            self.send_message("message", ("warning", "GPU不可用", "CUDA驱动版本不足，切换到CPU模式"))
                            # 切换到CPU模式并重新尝试加载
                            self.send_message("status", "CUDA驱动版本不足，切换到CPU模式...")
                            self.using_cpu = True
                            self.send_message("update_acceleration_button")
                            model_key = f"{model_size}_cpu_{model_path}"
                            self.model_cache[model_key] = WhisperModel(
                                model_path, device="cpu", compute_type="int8"
                            )
                            logger.info(f"已从本地加载模型到CPU: {model_path}")
                        else:
                            # 其他运行时错误
                            raise e
                else:
                    # 自动下载到程序目录
                    model_dir = get_model_dir()
                    os.makedirs(model_dir, exist_ok=True)

                    # 延迟创建requests会话
                    if self.requests_session is None:
                        self.requests_session = requests.Session()
                        try:
                            import certifi
                            self.requests_session.verify = certifi.where()
                            logger.info(f"使用certifi证书: {certifi.where()}")
                        except:
                            self.requests_session.verify = False
                            logger.warning("无法加载certifi，已禁用SSL验证")

                    try:
                        self.model_cache[model_key] = WhisperModel(
                            model_size,
                            device=device,
                            compute_type="int8",
                            download_root=str(model_dir),
                            resume_download=True,
                            session=self.requests_session
                        )
                        logger.info(f"已下载并加载模型: {model_size} 到 {model_dir}")
                    except RuntimeError as e:
                        if "CUDA driver version is insufficient" in str(e):
                            # 处理CUDA驱动版本不足的情况
                            logger.warning("CUDA驱动版本不足，尝试回退到CPU模式")
                            self.send_message("message", ("warning", "GPU不可用", "CUDA驱动版本不足，切换到CPU模式"))
                            # 切换到CPU模式并重新尝试加载
                            self.send_message("status", "CUDA驱动版本不足，切换到CPU模式...")
                            self.using_cpu = True
                            self.send_message("update_acceleration_button")
                            model_key = f"{model_size}_cpu_{model_path}"
                            self.model_cache[model_key] = WhisperModel(
                                model_size,
                                device="cpu",
                                compute_type="int8",
                                download_root=str(model_dir),
                                resume_download=True,
                                session=self.requests_session
                            )
                            logger.info(f"已下载并加载模型到CPU: {model_size}")
                        else:
                            # 其他运行时错误
                            raise e

            model = self.model_cache[model_key]

            # 更新状态
            self.send_message("status", "正在分析音频")
            self.send_message("progress", 20)
            logger.info("开始分析音频")

            # 执行转录
            supports_progress = False
            try:
                sig = inspect.signature(model.transcribe)
                supports_progress = 'progress_callback' in sig.parameters
                logger.info(f"faster-whisper版本支持进度回调: {supports_progress}")
            except Exception as e:
                logger.warning(f"无法检测进度回调支持: {e}，假设不支持")
                supports_progress = False

            if supports_progress:
                segments, info = model.transcribe(
                    self.filename,
                    beam_size=5,
                    language=language,
                    progress_callback=self.update_progress
                )
            else:
                segments, info = model.transcribe(
                    self.filename,
                    beam_size=5,
                    language=language
                )

                segments_list = list(segments)
                total_segments = len(segments_list)
                self.send_message("progress", 90)
                logger.info(f"音频分析完成，开始处理转录结果，共{total_segments}个片段")

                # 显示结果
                self.send_message("clear_result")

                # 显示语言检测结果
                if info.language and info.language_probability:
                    self.send_message("append_result",
                                      f"检测语言: {info.language} (置信度: {info.language_probability * 100:.1f}%)\n")
                    logger.info(f"检测语言: {info.language} (置信度: {info.language_probability * 100:.1f}%)")

                # 显示音频时长
                self.send_message("append_result", f"音频时长: {info.duration:.1f}秒\n\n")
                logger.info(f"音频时长: {info.duration:.1f}秒")

                # 创建结果目录
                output_dir = get_app_dir() / "transcripts"
                os.makedirs(output_dir, exist_ok=True)

                # 拼接完整文本
                full_text = ""
                for i, segment in enumerate(segments_list, 1):
                    text = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                    full_text += segment.text + "\n"

                    # 更新进度（处理结果阶段）
                    progress = 90 + (i / total_segments) * 10
                    self.send_message("progress", progress)

                    if i % 10 == 0 or i == total_segments:
                        logger.info(f"已处理 {i}/{total_segments} 个片段")

                # 保存结果到文件
                self.save_transcription_results(info, segments_list, output_dir)

                return

            # 处理进度回调不支持的情况
            segments_list = list(segments)
            total_segments = len(segments_list)

            try:
                # 更新状态
                self.send_message("status", "正在处理结果")
                self.send_message("progress", 90)
                logger.info(f"音频分析完成，开始处理转录结果，共{total_segments}个片段")

                # 显示结果
                self.send_message("clear_result")

                # 显示语言检测结果
                if info.language and info.language_probability:
                    self.send_message("append_result",
                                    f"检测语言: {info.language} (置信度: {info.language_probability * 100:.1f}%)\n")
                    logger.info(f"检测语言: {info.language} (置信度: {info.language_probability * 100:.1f}%)")

                # 显示音频时长
                self.send_message("append_result", f"音频时长: {info.duration:.1f}秒\n\n")
                logger.info(f"音频时长: {info.duration:.1f}秒")

                # 创建结果目录
                output_dir = get_app_dir() / "transcripts"
                os.makedirs(output_dir, exist_ok=True)

                # 保存结果到文件
                self.save_transcription_results(info, segments_list, output_dir)

            except Exception as e:
                logger.error(f"转录过程中发生错误: {e}")
                self.send_message("message", ("error", "错误", f"转录过程中发生错误: {e}"))
                self.send_message("status", "转录失败")
                raise

        except Exception as e:
            logger.error(f"转录过程中发生严重错误: {e}")
            self.send_message("message", ("error", "严重错误", f"转录过程中发生严重错误: {e}"))
            self.send_message("status", "严重错误")
            raise

        finally:
            # 恢复UI状态
            self.transcribe_btn.config(state=tk.NORMAL)
            self.file_btn.config(state=tk.NORMAL)
            self.model_size.config(state=tk.NORMAL)
            self.accelerate_btn.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.NORMAL)
            self.language.config(state=tk.NORMAL)

            # 释放资源
            if self.requests_session:
                self.requests_session.close()
                logger.info("已关闭requests会话")
            if hasattr(self, 'model_cache'):
                for model in self.model_cache.values():
                    del model
                logger.info("已清理模型缓存")

    def save_transcription_results(self, info, segments_list, output_dir):
        """保存转录结果到文件"""
        try:
            base_name = os.path.splitext(os.path.basename(self.filename))[0]
            saved_files = []

            # 保存为TXT文件
            if self.txt_var.get():
                txt_file = output_dir / f"{base_name}_transcript.txt"
                with open(txt_file, "w", encoding="utf-8") as f:
                    if info.language:
                        f.write(f"检测语言: {info.language}\n")
                    f.write(f"音频时长: {info.duration:.1f}秒\n\n")

                    for segment in segments_list:
                        f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")

                saved_files.append(txt_file)
                logger.info(f"已保存TXT文件: {txt_file}")

            # 保存为SRT文件
            if self.srt_var.get():
                srt_file = output_dir / f"{base_name}.srt"
                with open(srt_file, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(segments_list, 1):
                        f.write(f"{i}\n")
                        start_time = format_srt_time(segment.start)
                        end_time = format_srt_time(segment.end)
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{segment.text.strip()}\n\n")

                saved_files.append(srt_file)
                logger.info(f"已保存SRT文件: {srt_file}")

            # 完成提示
            message = "转录完成\n"
            for file in saved_files:
                message += f"已保存到: {file}\n"

            self.send_message("message", ("info", "完成", message))
            self.send_message("status", "完成! 结果已保存")
            self.send_message("progress", 100)
            logger.info("转录完成")

        except IOError as e:
            logger.error(f"文件保存失败: {e}")
            self.send_message("message", ("error", "错误", f"文件保存失败: {e}"))
            self.send_message("status", "文件保存失败")
            raise

    def send_message(self, message_type, message=None):
        self.message_queue.put((message_type, message))

    def process_messages(self):
        while not self.message_queue.empty():
            message_type, message = self.message_queue.get()
            if message_type == "status":
                self.status_var.set(message)
            elif message_type == "progress":
                self.progress_var.set(message)
            elif message_type == "message":
                icon, title, text = message
                messagebox.showinfo(title, text) if icon == "info" else messagebox.showerror(title, text)
            elif message_type == "clear_result":
                if self.result_frame_initialized:
                    self.result_text.delete(1.0, tk.END)
            elif message_type == "append_result":
                if self.result_frame_initialized:
                    self.result_text.insert(tk.END, message)
            elif message_type == "update_acceleration_button":
                self.toggle_acceleration()

        self.root.after(100, self.process_messages)

    def update_progress(self, progress):
        self.send_message("progress", progress * 100)

    def create_result_frame(self):
        if not self.result_frame_initialized:
            self.result_text = tk.Text(self.result_frame, font=self.font)
            self.result_text.pack(fill=tk.BOTH, expand=True)
            self.result_frame_initialized = True
        else:
            self.result_text.delete(1.0, tk.END)


def main():
    root = tk.Tk()
    app = WhisperTranscriber(root)
    root.mainloop()


if __name__ == "__main__":
    main()