from parser_model import *
import torch
from torch.nn import *
model = ParserModel(np.zeros((39638, 50), dtype=np.float32))
model.load_state_dict(torch.load("results/20210312_090655/model.weights"))
model.eval()
