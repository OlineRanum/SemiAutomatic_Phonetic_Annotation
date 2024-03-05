import omegaconf
from openhands.apis.classification_model import ClassificationModel
from openhands.core.exp_utils import get_trainer
import sys

cfg = omegaconf.OmegaConf.load("configs/gcn_train.yaml")
trainer = get_trainer(cfg)

model = ClassificationModel(cfg=cfg, trainer=trainer)
model.init_from_checkpoint_if_available()
model.fit()
