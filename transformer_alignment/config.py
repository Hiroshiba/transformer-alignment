from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from transformer_alignment.utility import dataclass_utility
from transformer_alignment.utility.git_utility import get_commit_id, get_branch_name


@dataclass
class DatasetConfig:
    input_glob: str
    target_glob: str
    num_test: int
    seed: int = 0


@dataclass
class TransformerNetworkConfig:
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = 'relu'


@dataclass
class NetworkConfig:
    in_size: int
    out_size: int
    transformer_network: TransformerNetworkConfig


@dataclass
class ModelConfig:
    pass


@dataclass
class TrainConfig:
    batchsize: int
    num_processes: Optional[int]
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any] = field(default=dict(
        name='Adam',
        betas=(0.9, 0.98),
        eps=1e-9,
    ))
    scheduler: Dict[str, Any] = field(default=dict(
        name='Noam',
        d_model=512,
        warmup_step=4000,
        scale=1.0,
    ))


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags['git-commit-id'] = get_commit_id()
        self.project.tags['git-branch-name'] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass
