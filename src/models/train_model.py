from TREQS.model import TREQS
from TREQS.training.params import params


model = TREQS(params)

params["task"] = "train"
model.train()
