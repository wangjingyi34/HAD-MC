"""Distillation Agent for HAD-MC 2.0"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import copy
import logging

logger = logging.getLogger(__name__)


class DistillationAgent:
    """
    Distillation Agent: Feature-aligned knowledge distillation.

    Transfers knowledge from a larger teacher model to a smaller student model.
    Uses:
    - Soft label distillation (KL divergence)
    - Hard label distillation (cross-entropy)
    - Feature alignment (matching intermediate representations)
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = 'cpu'
    ):
        """
        Initialize Distillation Agent.

        Args:
            teacher_model: Teacher model
            student_model: Student model
            device: Device to run on
        """
        self.teacher = teacher_model
        self.student = student_model
        self.device = device

        self.teacher.eval()
        self.teacher.to(self.device)

        # Distillation hyperparameters
        self.temperature = 4.0
        self.alpha = 0.5  # Weight for soft loss

    def get_action_space(self) -> Dict:
        """
        Get the action space for the distillation agent.

        Returns:
            dict: Action space definition (continuous for temperature and alpha)
        """
        return {
            'type': 'continuous',
            'temperature': (1.0, 20.0),
            'alpha': (0.0, 1.0),
        }

    def distill(
        self,
        train_loader,
        val_loader,
        temperature: Optional[float] = None,
        alpha: Optional[float] = None,
        epochs: int = 10,
        lr: float = 1e-4,
        feature_loss_weight: float = 0.1
    ) -> nn.Module:
        """
        Execute knowledge distillation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            temperature: Temperature for soft labels (optional)
            alpha: Weight for distillation loss (optional)
            epochs: Number of training epochs
            lr: Learning rate
            feature_loss_weight: Weight for feature alignment loss

        Returns:
            nn.Module: Distilled student model
        """
        if temperature is not None:
            self.temperature = temperature
        if alpha is not None:
            self.alpha = alpha

        self.student.train()
        self.student.to(self.device)

        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)

        best_val_acc = 0
        best_student_state = None

        logger.info(f"Starting distillation: T={self.temperature}, alpha={self.alpha}")

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_soft_loss = 0
            epoch_hard_loss = 0
            epoch_feature_loss = 0

            # Training
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                # Teacher output (no gradient)
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs)

                # Student output
                student_outputs = self.student(inputs)

                # Handle different output formats
                if isinstance(teacher_outputs, (tuple, list)):
                    teacher_outputs = teacher_outputs[0]
                if isinstance(student_outputs, (tuple, list)):
                    student_outputs = student_outputs[0]

                # ===== Soft label loss (distillation) =====
                # Use KL divergence with temperature scaling
                soft_loss = F.kl_div(
                    F.log_softmax(student_outputs / self.temperature, dim=1),
                    F.softmax(teacher_outputs / self.temperature, dim=1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)

                # ===== Hard label loss =====
                hard_loss = F.cross_entropy(student_outputs, targets)

                # ===== Feature alignment loss =====
                feature_loss = self._compute_feature_loss(inputs)

                # ===== Total loss =====
                loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss + feature_loss_weight * feature_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_soft_loss += soft_loss.item()
                epoch_hard_loss += hard_loss.item()
                epoch_feature_loss += feature_loss.item()

            # Validation
            val_acc = self._evaluate(self.student, val_loader)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Loss={epoch_loss / len(train_loader):.4f}, "
                f"Soft={epoch_soft_loss / len(train_loader):.4f}, "
                f"Hard={epoch_hard_loss / len(train_loader):.4f}, "
                f"Feature={epoch_feature_loss / len(train_loader):.4f}, "
                f"Val Acc={val_acc:.4f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_student_state = copy.deepcopy(self.student.state_dict())

        # Restore best model
        if best_student_state is not None:
            self.student.load_state_dict(best_student_state)
            logger.info(f"Restored best model with validation accuracy: {best_val_acc:.4f}")

        return self.student

    def _compute_feature_loss(
        self,
        inputs: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute feature alignment loss.

        Matches intermediate representations between teacher and student.

        Args:
            inputs: Input tensor
            layer_indices: List of layer indices to align (optional)

        Returns:
            torch.Tensor: Feature alignment loss
        """
        with torch.no_grad():
            teacher_features = self._extract_features(self.teacher, inputs, layer_indices)

        student_features = self._extract_features(self.student, inputs, layer_indices)

        # Compute MSE loss between teacher and student features
        if not teacher_features or not student_features:
            return torch.tensor(0.0, device=self.device)

        feature_loss = 0
        for t_feat, s_feat in zip(teacher_features, student_features):
            # Adaptive pooling if dimensions don't match
            if t_feat.shape != s_feat.shape:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])

            feature_loss += F.mse_loss(s_feat, t_feat)

        feature_loss = feature_loss / len(teacher_features)

        return feature_loss

    def _extract_features(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Extract intermediate features from a model.

        Args:
            model: PyTorch model
            inputs: Input tensor
            layer_indices: List of layer indices to extract (optional)

        Returns:
            list: List of feature tensors
        """
        features = []
        hooks = []

        def make_hook():
            def hook(module, input, output):
                features.append(output.clone())
            return hook
        return make_hook

        # Register hooks on selected layers
        conv_layers = []
        for idx, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, nn.Conv2d):
                conv_layers.append((idx, name, module))

        # Select layers to extract from
        if layer_indices is None:
            # Extract from middle layers
            num_conv = len(conv_layers)
            selected_indices = [num_conv // 4, num_conv // 2, 3 * num_conv // 4]
        else:
            selected_indices = layer_indices

        for idx in selected_indices:
            if idx < len(conv_layers):
                name = conv_layers[idx][1]
                handle = conv_layers[idx][2].register_forward_hook(make_hook())
                hooks.append((name, handle))

        # Forward pass
        with torch.no_grad():
            _ = model(inputs)

        # Remove hooks
        for name, handle in hooks:
            handle.remove()

        return features

    def _evaluate(self, model: nn.Module, dataloader) -> float:
        """
        Evaluate model accuracy.

        Args:
            model: PyTorch model
            dataloader: Data loader

        Returns:
            float: Accuracy
        """
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)

                # Handle different output formats
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # Get predictions
                if outputs.dim() > 1 and outputs.shape[1] > 1:
                    _, predicted = outputs.max(1)
                else:
                    predicted = (outputs > 0).long()

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def get_state(self) -> Dict:
        """
        Get the current state of the distillation agent.

        Returns:
            dict: Current state information
        """
        total_student_params = sum(p.numel() for p in self.student.parameters())
        total_teacher_params = sum(p.numel() for p in self.teacher.parameters())

        return {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'student_params': total_student_params,
            'teacher_params': total_teacher_params,
            'compression_ratio': total_teacher_params / total_student_params if total_student_params > 0 else 0,
        }

    def get_action(
        self,
        state: torch.Tensor
    ) -> tuple:
        """
        Get a distillation action (sample from action space).

        Args:
            state: Current state tensor

        Returns:
            tuple: (distillation_config, log_prob)
        """
        # For this implementation, return a random action
        # In the full MARL setting, this would come from the PPO policy
        import random

        temperature = random.uniform(1.0, 20.0)
        alpha = random.uniform(0.0, 1.0)

        distillation_config = {
            'temperature': temperature,
            'alpha': alpha,
        }

        # Log prob would be computed by PPO
        log_prob = 0.0

        return distillation_config, log_prob

    def apply_action(
        self,
        model: nn.Module,
        action: Dict
    ) -> nn.Module:
        """
        Apply a distillation action to the model.

        Args:
            model: PyTorch model (student)
            action: Action dictionary {'temperature': float, 'alpha': float}

        Returns:
            nn.Module: Model with updated distillation parameters
        """
        self.temperature = action['temperature']
        self.alpha = action['alpha']

        # Distillation requires training, so we just return the model
        # The actual distillation is called separately
        return model

    def __repr__(self) -> str:
        student_params = sum(p.numel() for p in self.student.parameters())
        teacher_params = sum(p.numel() for p in self.teacher.parameters())

        return (f"DistillationAgent(T={self.temperature}, alpha={self.alpha}, "
                f"student_params={student_params}, teacher_params={teacher_params})")
