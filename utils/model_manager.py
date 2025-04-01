import os
import json
import requests
import shutil
from typing import Dict, Optional, Any
from tqdm import tqdm
from utils.config import config
from utils.logger import logger

class ModelManager:
    """模型管理工具"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.model_info_file = "models/model_info.json"
            self._load_model_info()
    
    def _load_model_info(self):
        """加载模型信息"""
        try:
            if os.path.exists(self.model_info_file):
                with open(self.model_info_file, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
        except Exception as e:
            logger.exception(f"加载模型信息失败: {str(e)}")
            self.model_info = {}
    
    def get_model_info(self, engine_type: str, engine_name: str, model_name: str=None) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        try:

            if model_name == None:
                return self.model_info[engine_type][engine_name]
            return self.model_info[engine_type][engine_name][model_name]
        except KeyError:
            return None
    
    def is_model_downloaded(self, engine_type: str, engine_name: str, model_name: str) -> bool:
        """检查模型是否已下载
        
        Args:
            engine_type: 引擎类型 (asr/tts/translation)
            engine_name: 引擎名称 (whisper/faster_whisper/vosk等)
            model_name: 模型名称 (base/small等)
            
        Returns:
            bool: 模型是否已正确下载
        """
        try:
            # 获取模型信息
            model_info = self.get_model_info(engine_type, engine_name, model_name)
            if not model_info:
                logger.exception(f"未找到模型信息: {engine_type}/{engine_name}/{model_name}")
                return False

            model_path  = model_info["save_path"]

            # 检查是否是多文件模型
            if "files" in model_info:
                # 检查每个必需文件是否存在
                for file_info in model_info["files"]:
                    file_path = os.path.join(model_path, file_info["name"])
                    if not os.path.exists(file_path):
                        return False
                return True
            
            return False
            
        except Exception as e:
            logger.exception(f"检查模型下载状态失败: {str(e)}")
            return False
    
    def download_model(self, engine_type: str, engine_name: str, model_name: str, progress_callback=None) -> bool:
        """下载模型"""
        try:
            # 获取模型信息
            model_info = self.get_model_info(engine_type, engine_name, model_name)
            if not model_info:
                logger.exception(f"未找到模型信息: {engine_type}/{engine_name}/{model_name}")
                return False
            
            model_path = model_info["save_path"]
            
            os.makedirs(model_path, exist_ok=True)
            
            # 检查是否是多文件模型
            if "files" in model_info:
                total_size = sum(f["size"] for f in model_info["files"])
                downloaded_size = 0
                
                # 下载每个文件
                for file_info in model_info["files"]:
                    file_url = file_info["url"]
                    file_name = file_info["name"]
                    file_size = file_info["size"]
                    
                    # 临时文件路径
                    temp_file = os.path.join(model_path, f"{file_name}.tmp")
                    final_file = os.path.join(model_path, file_name)
                    
                    # 如果文件已存在且大小正确，跳过下载
                    if os.path.exists(final_file) and os.path.getsize(final_file) == file_size:
                        downloaded_size += file_size
                        if progress_callback:
                            progress_callback(min(1.0, downloaded_size / total_size))
                        continue
                    
                    # 下载文件
                    if not self._download_file(file_url, temp_file, final_file, file_size,
                                            lambda p: progress_callback(min(1.0, (downloaded_size + p * file_size) / total_size)) if progress_callback else None):
                        return False
                    
                    downloaded_size += file_size
                
                return True
                
            else:
                # 单文件模型的原有逻辑
                url = model_info['url']
                is_zip = url.endswith('.zip')
                temp_file = os.path.join(model_path, "model.zip.tmp" if is_zip else "model.bin.tmp")
                final_file = os.path.join(model_path, "model.zip" if is_zip else "model.bin")
                
                return self._download_file(url, temp_file, final_file, model_info.get('size', 0), progress_callback)
                
        except Exception as e:
            logger.exception(f"模型下载失败: {str(e)}")
            return False
    
    def _download_file(self, url: str, temp_file: str, final_file: str, expected_size: int, progress_callback=None) -> bool:
        """下载单个文件"""
        try:
            # 设置请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            }
            
            # 创建会话
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=10,
                max_retries=3,
                pool_block=False
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            try:
                # 首先发送HEAD请求获取文件大小
                response = session.head(url, headers=headers)
                response.raise_for_status()
                
                # 从响应头获取文件大小
                content_length = response.headers.get('content-length')
                if content_length:
                    total_size = int(content_length)
                else:
                    total_size = expected_size
                
                # 检查是否支持断点续传
                accept_ranges = response.headers.get('accept-ranges', '').lower()
                supports_resume = accept_ranges == 'bytes'
                
                # 获取已下载的大小
                downloaded_size = 0
                if os.path.exists(temp_file):
                    downloaded_size = os.path.getsize(temp_file)
                
                # 如果文件已存在且大小匹配,说明已下载完成
                if total_size and downloaded_size == total_size:
                    logger.info("文件已下载完成")
                    if progress_callback:
                        progress_callback(1.0)
                    if not os.path.exists(final_file):
                        os.rename(temp_file, final_file)
                    return True
                
                # 如果不支持断点续传或文件大小不匹配,删除临时文件重新下载
                if not supports_resume or (total_size and downloaded_size > total_size):
                    logger.info("不支持断点续传或文件大小不匹配,重新开始下载")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    downloaded_size = 0
                
                # 设置Range头
                if supports_resume and downloaded_size > 0:
                    headers['Range'] = f'bytes={downloaded_size}-'
                
                # 开始下载
                logger.info(f"开始下载: {url}")
                response = session.get(url, headers=headers, stream=True)
                response.raise_for_status()
                
                # 使用tqdm显示进度
                with tqdm(total=total_size, initial=downloaded_size, unit='B', unit_scale=True) as pbar:
                    with open(temp_file, 'ab' if downloaded_size > 0 else 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                f.flush()
                                downloaded_size += len(chunk)
                                pbar.update(len(chunk))
                                
                                # 更新进度
                                if progress_callback and total_size:
                                    progress = min(1.0, downloaded_size / total_size)
                                    progress_callback(progress)
                
                # 下载完成,重命名文件
                if os.path.exists(final_file):
                    os.remove(final_file)
                os.rename(temp_file, final_file)
                
                # 如果是zip文件，解压
                if final_file.endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(final_file, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(final_file))
                    os.remove(final_file)
                
                logger.info(f"文件下载完成: {os.path.basename(final_file)}")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.exception(f"下载请求失败: {str(e)}")
                return False
            finally:
                session.close()
                
        except Exception as e:
            logger.exception(f"文件下载失败: {str(e)}")
            return False
    
    def get_download_progress(self, engine_type: str, engine_name: str, model_name: str) -> float:
        """获取下载进度"""
        try:
            model_info = self.get_model_info(engine_type, engine_name, model_name)
            if not model_info:
                return 0.0
            
            # 如果是 edge-tts，直接返回 1.0
            if engine_type == "tts" and engine_name == "edge_tts":
                return 1.0
            
            model_dir = config.get_model_dir(engine_type)
            if engine_type == "asr":
                model_path = os.path.join(model_dir, engine_name, model_name)
            elif engine_type == "translation":
                model_path = os.path.join(model_dir, engine_name, model_name)
            elif engine_type == "tts":
                model_path = os.path.join(model_dir, engine_name, model_name)
            else:
                return 0.0
                
            # 检查临时文件和最终文件
            temp_file = os.path.join(model_path, "model.bin.tmp")
            final_file = os.path.join(model_path, "model.bin")
            
            # 如果最终文件存在，说明下载完成
            if os.path.exists(final_file):
                return 1.0
            
            # 如果临时文件不存在，说明还未开始下载
            if not os.path.exists(temp_file):
                return 0.0
            
            # 获取临时文件大小
            current_size = os.path.getsize(temp_file)
            
            # 尝试从 HEAD 请求获取总大小
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.head(model_info['url'], headers=headers, timeout=5)
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    if total_size > 0:
                        return min(1.0, current_size / total_size)
            except Exception as e:
                logger.warning(f"获取文件大小失败: {str(e)}")
            
            # 如果无法获取实际大小，使用配置中的预估大小
            total_size = model_info.get('size', 0)
            if total_size > 0:
                return min(1.0, current_size / total_size)
            
            return 0.0
            
        except Exception as e:
            logger.exception(f"获取下载进度失败: {str(e)}")
            return 0.0
    
    def delete_model(self, engine_type: str, engine_name: str, model_name: str) -> bool:
        """删除模型"""
        try:
            model_dir = config.get_model_dir(engine_type)
            if engine_type == "asr":
                model_path = os.path.join(model_dir, engine_name, model_name)
            elif engine_type == "translation":
                model_path = os.path.join(model_dir, engine_name, model_name)
            elif engine_type == "tts":
                model_path = os.path.join(model_dir, engine_name, model_name)
            else:
                return False
            
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                logger.info(f"已删除模型: {engine_type}/{engine_name}/{model_name}")
                return True
            return False
            
        except Exception as e:
            logger.exception(f"删除模型失败: {str(e)}")
            return False

# 创建全局模型管理器实例
model_manager = ModelManager() 