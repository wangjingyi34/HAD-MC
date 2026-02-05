"""Update Agent for HAD-MC 2.0"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import copy
import logging
import hashlib

logger = logging.getLogger(__name__)


class UpdateAgent:
    """
    Update Agent: Incremental model updates for edge deployment.

    Supports three update strategies:
    1. Full update: Complete model download
    2. Incremental update: Update selected layers only
    3. Hash-based update: Delta update based on weight hashing

    Enables efficient cloud-edge synchronization.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        """
        Initialize Update Agent.

        Args:
            model: PyTorch model
            device: Device to run on
        """
        self.model = model
        self.device = device

        # Update strategies
        self.strategies = ['full', 'incremental', 'hash_based']
        self.update_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Hash tables for delta updates
        self.hash_tables = {}
        self.weight_hashes = {}

    def get_action_space(self) -> Dict:
        """
        Get the action space for the update agent.

        Returns:
            dict: Action space definition
        """
        return {
            'type': 'discrete',
            'strategy': self.strategies,
            'update_ratio': self.update_ratios,
        }

    def build_hash_tables(self, num_clusters: int = 256):
        """
        Build weight hash tables for delta updates.

        Clusters weights and stores centroids for efficient updates.

        Args:
            num_clusters: Number of clusters for K-means
        """
        logger.info(f"Building hash tables with {num_clusters} clusters...")

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data.flatten().cpu()

                # Simplified K-means using random centroids
                # (Full implementation would use sklearn or similar)
                centroids = self._initialize_centroids(weight, num_clusters)

                self.hash_tables[name] = {
                    'centroids': centroids,
                    'num_clusters': num_clusters,
                }

                # Compute initial hash
                self.weight_hashes[name] = self._compute_hash(weight)

        logger.info(f"Built hash tables for {len(self.hash_tables)} layers")

    def _initialize_centroids(self, weight: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """
        Initialize cluster centroids (simplified K-means).

        Args:
            weight: Weight tensor
            num_clusters: Number of clusters

        Returns:
            torch.Tensor: Cluster centroids
        """
        # Random initialization
        indices = torch.randperm(weight.numel())[:num_clusters]
        centroids = weight[indices]
        return centroids

    def _compute_hash(self, tensor: torch.Tensor) -> str:
        """
        Compute SHA256 hash of a tensor.

        Args:
            tensor: Input tensor

        Returns:
            str: Hash string
        """
        return hashlib.sha256(
            tensor.numpy().tobytes()
        ).hexdigest()

    def update(
        self,
        new_data_loader,
        update_strategy: Optional[str] = None,
        update_ratio: Optional[float] = None,
        epochs: int = 5,
        lr: float = 1e-4
    ) -> nn.Module:
        """
        Execute model update using specified strategy.

        Args:
            new_data_loader: New data for update
            update_strategy: Update strategy ('full', 'incremental', 'hash_based')
            update_ratio: Ratio of layers to update (for incremental)
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            nn.Module: Updated model
        """
        if update_strategy is None:
            update_strategy = 'full'
        if update_ratio is None:
            update_ratio = 1.0

        logger.info(f"Starting {update_strategy} update (ratio={update_ratio})")

        if update_strategy == 'full':
            return self._full_update(new_data_loader, epochs, lr)
        elif update_strategy == 'incremental':
            return self._incremental_update(new_data_loader, update_ratio, epochs, lr)
        elif update_strategy == 'hash_based':
            return self._hash_based_update(new_data_loader, epochs, lr)
        else:
            logger.warning(f"Unknown strategy: {update_strategy}, using full update")
            return self._full_update(new_data_loader, epochs, lr)

    def _full_update(
        self,
        new_data_loader,
        epochs: int,
        lr: float
    ) -> nn.Module:
        """
        Full model update: retrain entire model.

        Args:
            new_data_loader: New data
            epochs: Number of epochs
            lr: Learning rate

        Returns:
            nn.Module: Updated model
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        total_params = sum(p.numel() for p in self.model.parameters())

        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in new_data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)

                # Handle different output formats
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logger.info(f"Full update - Epoch {epoch + 1}/{epochs}: "
                       f"Loss={epoch_loss / len(new_data_loader):.4f}")

        updated_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Full update: {updated_params} parameters ({updated_params/total_params:.1%} of original)")

        return self.model

    def _incremental_update(
        self,
        new_data_loader,
        update_ratio: float,
        epochs: int,
        lr: float
    ) -> nn.Module:
        """
        Incremental update: update only specified ratio of layers.

        Args:
            new_data_loader: New data
            update_ratio: Ratio of layers to update [0, 1]
            epochs: Number of epochs
            lr: Learning rate

        Returns:
            nn.Module: Updated model
        """
        self.model.train()

        # Get all updatable layers
        updatable_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Extract layer name from parameter name
                layer_name = '.'.join(name.split('.')[:-1])
                if layer_name not in updatable_layers:
                    updatable_layers.append(layer_name)

        # Select layers to update (last 'update_ratio' fraction)
        num_update = int(len(updatable_layers) * update_ratio)
        layers_to_update = updatable_layers[-num_update:] if num_update > 0 else []

        logger.info(f"Incremental update: updating {len(layers_to_update)}/{len(updatable_layers)} layers")

        # Freeze layers not to update
        for name, param in self.model.named_parameters():
            layer_name = '.'.join(name.split('.')[:-1])
            if layer_name not in layers_to_update:
                param.requires_grad = False

        # Create optimizer for unfrozen parameters
        params_to_update = [
            param for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        optimizer = torch.optim.Adam(params_to_update, lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in new_data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)

                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logger.info(f"Incremental update - Epoch {epoch + 1}/{epochs}: "
                       f"Loss={epoch_loss / len(new_data_loader):.4f}")

        # Restore requires_grad for all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        return self.model

    def _hash_based_update(
        self,
        new_data_loader,
        epochs: int,
        lr: float
    ) -> nn.Module:
        """
        Hash-based update: update only changed weight clusters.

        Args:
            new_data_loader: New data
            epochs: Number of epochs
            lr: Learning rate

        Returns:
            nn.Module: Updated model
        """
        self.model.train()

        # Ensure hash tables are built
        if not self.hash_tables:
            self.build_hash_tables()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        num_updates = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in new_data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)

                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                loss = F.cross_entropy(outputs, targets)
                loss.backward()

                # Check which layers had significant updates
                with torch.no_grad():
                    for name, module in self.model.named_modules():
                        if isinstance(module, (nn.Conv2d, nn.Linear)):
                            current_hash = self._compute_hash(module.weight.data)

                            if name in self.weight_hashes:
                                if current_hash != self.weight_hashes[name]:
                                    # Layer changed significantly
                                    num_updates += 1
                                    self.weight_hashes[name] = current_hash

                optimizer.step()
                epoch_loss += loss.item()

            logger.info(f"Hash-based update - Epoch {epoch + 1}/{epochs}: "
                       f"Loss={epoch_loss / len(new_data_loader):.4f}, "
                       f"Layers updated: {num_updates}")

        logger.info(f"Hash-based update: {num_updates} layers with significant changes")

        return self.model

    def compute_delta(self, old_model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute delta (difference) between current and old model.

        Used for efficient incremental updates.

        Args:
            old_model: Previous model version

        Returns:
            dict: Delta parameters for each layer
        """
        delta = {}

        for (name, param), (old_name, old_param) in zip(
            self.model.named_parameters(),
            old_model.named_parameters()
        ):
            if name == old_name:
                delta[name] = param.data - old_param.data

        return delta

    def apply_delta(self, delta: Dict[str, torch.Tensor]):
        """
        Apply delta update to model.

        Args:
            delta: Dictionary of parameter deltas
        """
        for name, param in self.model.named_parameters():
            if name in delta:
                param.data += delta[name].to(self.device)

        logger.info(f"Applied delta updates to {len(delta)} parameters")

    def get_update_size(self, delta: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """
        Calculate size of update (in MB).

        Args:
            delta: Delta dictionary (optional, computes from model if None)

        Returns:
            float: Update size in MB
        """
        if delta is None:
            # Estimate based on model
            size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        else:
            size = sum(v.numel() * v.element_size() for v in delta.values())

        return size / (1024 * 1024)  # Convert to MB

    def get_state(self) -> Dict:
        """
        Get the current state of the update agent.

        Returns:
            dict: Current state information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size = total_params * 4 / (1024 * 1024)  # MB (FP32)

        return {
            'total_params': total_params,
            'model_size_mb': model_size,
            'hash_tables_built': len(self.hash_tables) > 0,
            'num_hash_tables': len(self.hash_tables),
        }

    def get_action(
        self,
        state: torch.Tensor
    ) -> tuple:
        """
        Get an update action (sample from action space).

        Args:
            state: Current state tensor

        Returns:
            tuple: (update_config, log_prob)
        """
        # For this implementation, return a random action
        # In the full MARL setting, this would come from the PPO policy
        import random

        strategy = random.choice(self.strategies)
        update_ratio = random.choice(self.update_ratios)

        update_config = {
            'strategy': strategy,
            'update_ratio': update_ratio,
        }

        # Log prob would be computed by PPO
        log_prob = 0.0

        return update_config, log_prob

    def apply_action(
        self,
        model: nn.Module,
        action: Dict
    ) -> nn.Module:
        """
        Apply an update action to the model.

        Args:
            model: PyTorch model
            action: Action dictionary {'strategy': str, 'update_ratio': float}

        Returns:
            nn.Module: Model with update applied
        """
        self.model = model
        # Update agent now references the provided model

        # The actual update is triggered later with data
        return model

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.model.parameters())
        return (f"UpdateAgent(strategies={self.strategies}, "
                f"model_params={total_params}, "
                f"hash_tables={len(self.hash_tables)})")
