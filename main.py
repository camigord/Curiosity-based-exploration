import numpy as np

# custom modules
from utils.options import Options
from core.agents.a3c   import A3CAgent
from core.model import A3CCuriosity
from core.env_temp import DoomEnv

# 0. setting up
opt = Options()
np.random.seed(opt.seed)

agent = A3CAgent(opt.agent_params,env_prototype    = DoomEnv,
                                  model_prototype  = A3CCuriosity)
# 4. fit model
if opt.mode == 1:   # train
    agent.fit_model()
elif opt.mode == 2: # test opt.model_file
    agent.test_model()
