from omegaconf import DictConfig

from worker_agent import WorkerAgent
from worker_task import WorkerTask

from friday_server.agent import create_agent_server


server_cfg = DictConfig({})

agent = WorkerAgent(cfg=DictConfig())
agent.register_task_class(WorkerTask)


app = create_agent_server(server_cfg, agent)

app.run()
