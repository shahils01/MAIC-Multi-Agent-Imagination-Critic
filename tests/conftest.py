import pytest
import torch

from maic import MAIC, MAICConfig
from maic.demo import synthetic_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(31)


@pytest.fixture
def config():
    return MAICConfig(state_dim=2, agent_task_dim=2, d_model=16, num_heads=4,
                      spatial_layers=1, temporal_layers=1, decoder_layers=1,
                      world_layers=1, history_length=4)


@pytest.fixture
def model(config):
    return MAIC(config).eval()


@pytest.fixture
def batch(config):
    return synthetic_batch(config, batch_size=2, length=4, num_agents=3, seed=3)
