from script_utils import ScriptUtils

ScriptUtils.setup_script()

from data.data import RawDataset
from data.loaders.quality_debates_loader import QualityDebatesLoader
from train.train_utils import TrainUtils

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_model_run_script_config(args)

config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)
quality_debates_dataset = QualityDebatesLoader.load(full_dataset_filepath=script_config.full_dataset_filepath)

trainer = TrainUtils.get_trainer(config=config, raw_dataset=quality_debates_dataset, is_local=args.local)
if not args.load_only:
    trainer.train()
trainer.save_model()
