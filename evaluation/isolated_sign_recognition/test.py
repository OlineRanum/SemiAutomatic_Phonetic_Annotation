import sys 
import omegaconf
from openhands.apis.inference import InferenceModel


if len(sys.argv) != 2:
    print("Usage: python script.py <path_to_gcn_test_yaml>")
    sys.exit(1)
config_path = sys.argv[1]




cfg = omegaconf.OmegaConf.load(config_path)
model = InferenceModel(cfg=cfg)
model.init_from_checkpoint_if_available()
if cfg.data.test_pipeline.dataset.inference_mode:
    model.test_inference()
else:
    model.compute_test_accuracy()
