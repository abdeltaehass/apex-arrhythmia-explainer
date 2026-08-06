"""Phase 20 — federated learning simulation over PTB-XL device shards."""

from src.federated.fedavg import ClientUpdate, RoundRecord, average_state_dicts, local_train
from src.federated.partition import Client, build_clients, label_skew, partition_summary

__all__ = [
    "Client",
    "ClientUpdate",
    "RoundRecord",
    "average_state_dicts",
    "build_clients",
    "label_skew",
    "local_train",
    "partition_summary",
]
