import json
import app
info = getattr(app, 'MODEL_INFO', None)
if info is None:
    print('MODEL_INFO: None')
else:
    out = {
        'model_type': info.get('model_type'),
        'state_dict_path': info.get('state_dict_path'),
        'optimal_threshold': info.get('optimal_threshold')
    }
    print('MODEL_INFO:', json.dumps(out, indent=2))
