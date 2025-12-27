import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage configuration from YAML/JSON files"""

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"

    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        """Load config from YAML file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_json(path: Path) -> Dict[str, Any]:
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def load(path: Optional[Path] = None) -> Dict[str, Any]:
        """Load config from file, auto-detect format"""
        if path is None:
            return ConfigLoader.load_yaml(ConfigLoader.DEFAULT_CONFIG_PATH)

        path = Path(path)
        if path.suffix in ['.yaml', '.yml']:
            return ConfigLoader.load_yaml(path)
        elif path.suffix == '.json':
            return ConfigLoader.load_json(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    @staticmethod
    def save_yaml(config: Dict[str, Any], path: Path):
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def save_json(config: Dict[str, Any], path: Path):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge override config into base config (deep merge)"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def update_from_cli_args(config: Dict[str, Any], args) -> Dict[str, Any]:
        """Update config with common CLI argument overrides"""
        # Map CLI args to config paths
        cli_to_config = {
            'hidden_size': ('lstm', 'hidden_size'),
            'learning_rate': ('ppo', 'learning_rate'),
            'entropy_coef': ('ppo', 'entropy_coef'),
            'num_envs': ('training', 'num_envs'),
            'num_steps': ('training', 'num_steps'),
            'total_timesteps': ('training', 'total_timesteps'),
            'seq_len': ('ppo', 'seq_len'),
            'batch_size': ('ppo', 'batch_size'),
            'num_epochs': ('ppo', 'num_epochs'),
            'gamma': ('ppo', 'gamma'),
            'gae_lambda': ('ppo', 'gae_lambda'),
            'clip_epsilon': ('ppo', 'clip_epsilon'),
            'value_coef': ('ppo', 'value_coef'),
            'max_grad_norm': ('ppo', 'max_grad_norm'),
        }

        for arg_name, (section, key) in cli_to_config.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    config[section][key] = value

        return config
