import os


class ConfigManager:
    def __init__(self, file_path: str = None):
        # 获取项目根目录的绝对路径
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_file_dir)  # 只向上退一级到knowledge-base-builder

        # 如果没有指定配置文件路径，使用默认路径
        if file_path is None:
            file_path = os.path.join(self.project_root, 'resources', 'config.properties')

        self._props = {}
        self._load_properties(file_path)

        # 设置属性，并将相对路径转换为绝对路径
        for key, value in self._props.items():
            if key.endswith('_path') and not os.path.isabs(value):
                # 将相对路径转换为基于项目根目录的绝对路径
                value = os.path.join(self.project_root, value)
            setattr(self, key, value)

    def _load_properties(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', '!', ';')):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                elif ':' in line:
                    k, v = line.split(':', 1)
                else:
                    continue
                self._props[k.strip()] = v.strip()


config = ConfigManager()
