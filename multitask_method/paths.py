from pathlib import Path

base_data_input_dir = Path('')
base_prediction_dir = Path('')
base_log_dir = Path('')

ensemble_dir = 'ensemble'


def exp_log_dir(exp_name, fold):
    return base_log_dir / exp_name / str(fold)
