from typing import Any, Union, Dict
import importlib

class ConfigurationError(Exception):
    pass

def instantiate_from_config(config: Union[Dict[str, Any], Any]) -> Any:
    """
    Recursively instantiate objects from config dictionary
    
    Args:
        config: Dictionary containing __type__ and parameters, 
               or a primitive value
    
    Returns:
        Instantiated object with all nested objects
    """
    # If config is not a dict, return it as is (handles primitive values)
    if not isinstance(config, dict):
        return config
    
    # If no __type__, process dict values recursively
    if '__type__' not in config:
        return {
            key: instantiate_from_config(value)
            for key, value in config.items()
        }
    
    # Get the type path and remove it from config
    type_path = config.pop('__type__')
    
    # Process all nested configurations recursively
    processed_config = {
        key: instantiate_from_config(value)
        for key, value in config.items()
    }
    
    try:
        # Import the class
        module_path, class_name = type_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        target_class = getattr(module, class_name)
        
        # Create instance with processed config
        instance = target_class(**processed_config)
        return instance
    
    except (ImportError, AttributeError) as e:
        raise ConfigurationError(f"Failed to import {type_path}: {str(e)}")
    except Exception as e:
        raise ConfigurationError(f"Failed to instantiate {type_path}: {str(e)}")

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]