import os


class ConfigManager:
    def __init__(self, file_path: str = None):
        # Get the absolute path of the project root directory
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_file_dir)  # Go up one level to knowledge-base-builder

        # If no config file path is specified, use the default path
        if file_path is None:
            file_path = os.path.join(self.project_root, 'resources', 'config.properties')

        self._props = {}
        self._load_properties(file_path)

        # Set attributes and convert relative paths to absolute paths
        for key, value in self._props.items():
            if key.endswith('_path') and not os.path.isabs(value):
                # Convert relative paths to absolute paths based on project root directory
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
