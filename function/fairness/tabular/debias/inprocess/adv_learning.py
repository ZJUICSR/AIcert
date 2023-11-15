import torch
import torch.nn as nn
from fairness_datasets import FairnessDataset
from models.models import Net
import math
import numpy as np
import copy
from . import Classifier
import torch.optim as optim


class AdversarialDebiasing(nn.Module):
    def __init__(self, unprivileged_groups, privileged_groups, seed=None,
                 adversary_loss_weight=0.1, num_epochs=50, batch_size=128,
                 classifier_num_hidden_units=200, debias=True):
        super(AdversarialDebiasing, self).__init__()

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
            raise ValueError("Only one unprivileged_group or privileged_group supported.")
        self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]

        self.seed = seed
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias

        self.features_dim = None
        self.classifier = None
        self.adversary = None

    def _classifier_model(self, features, keep_prob):
        classifier = nn.Sequential(
            nn.Linear(self.features_dim, self.classifier_num_hidden_units),
            nn.ReLU(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.classifier_num_hidden_units, 1),
            nn.Sigmoid()
        )
        return classifier(features)

    def _adversary_model(self, pred_logits, true_labels):
        c = torch.nn.Parameter(torch.Tensor([1.0]))
        s = torch.sigmoid((1 + torch.abs(c)) * pred_logits)
        adversary = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        return adversary(torch.cat([s, s * true_labels, s * (1.0 - true_labels)], dim=1))

    def forward(self, features, true_labels, protected_attributes, keep_prob):
        pred_logits = self.classifier(features)
        pred_labels_loss = nn.BCEWithLogitsLoss()(pred_logits, true_labels)

        if self.debias:
            pred_protected_attributes_logits = self.adversary(pred_logits, true_labels)
            pred_protected_attributes_loss = nn.BCEWithLogitsLoss()(pred_protected_attributes_logits, protected_attributes)

        return pred_labels_loss, pred_protected_attributes_loss if self.debias else pred_labels_loss

    def fit(self, dataset):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        num_train_samples, self.features_dim = dataset.features.shape
        self.classifier = self._classifier_model(self.features_dim, keep_prob=0.8)
        if self.debias:
            self.adversary = self._adversary_model(None, None)

        optimizer_cls = optim.Adam(self.classifier.parameters(), lr=0.001)
        optimizer_adv = optim.Adam(self.adversary.parameters(), lr=0.001) if self.debias else None

        for epoch in range(self.num_epochs):
            shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
            for i in range(num_train_samples // self.batch_size):
                batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                batch_features = torch.tensor(dataset.features[batch_ids], dtype=torch.float32)
                batch_labels = torch.tensor(dataset.labels[batch_ids], dtype=torch.float32)
                batch_protected_attributes = torch.tensor(
                    dataset.protected_attributes[batch_ids][:,
                    dataset.protected_attribute_names.index(self.protected_attribute_name)],
                    dtype=torch.float32)

                optimizer_cls.zero_grad()
                if self.debias:
                    optimizer_adv.zero_grad()

                pred_labels_loss, pred_protected_attributes_loss = self.forward(
                    batch_features, batch_labels, batch_protected_attributes, keep_prob=0.8)

                pred_labels_loss.backward()
                optimizer_cls.step()

                if self.debias:
                    pred_protected_attributes_loss.backward()
                    optimizer_adv.step()

                if i % 200 == 0:
                    print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" %
                          (epoch, i, pred_labels_loss.item(), pred_protected_attributes_loss.item()))

        return self

    def predict(self, dataset):
        self.eval()
        num_test_samples, _ = dataset.features.shape
        batch_size = self.batch_size

        samples_covered = 0
        pred_labels = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = torch.tensor(dataset.features[batch_ids], dtype=torch.float32)
            batch_labels = torch.tensor(dataset.labels[batch_ids], dtype=torch.float32)
            batch_protected_attributes = torch.tensor(
                dataset.protected_attributes[batch_ids][:,
                dataset.protected_attribute_names.index(self.protected_attribute_name)],
                dtype=torch.float32)

            pred_labels_batch = self.classifier(batch_features).detach().numpy()
            pred_labels += pred_labels_batch[:, 0].tolist()
            samples_covered += len(batch_features)

        dataset_new = dataset.copy(deepcopy=True)
        dataset_new.scores = np.array(pred_labels, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(pred_labels) > 0.5).astype(np.float64).reshape(-1, 1)

        return dataset_new


class AdvNet(nn.Module):
    def __init__(self, input_shape, keep_prob):
        super().__init__()
        self.classifier = Net(input_shape=input_shape)
        self.adversary = self._adversary_model()
        
    def _classifier_model(self, features, keep_prob):
        classifier = nn.Sequential(
            nn.Linear(self.features_dim, self.classifier_num_hidden_units),
            nn.ReLU(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.classifier_num_hidden_units, 1),
            nn.Sigmoid()
        )
        return classifier(features)

    def _adversary_model(self, pred_logits, true_labels):
        c = torch.nn.Parameter(torch.Tensor([1.0]))
        s = torch.sigmoid((1 + torch.abs(c)) * pred_logits)
        adversary = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        return adversary(torch.cat([s, s * true_labels, s * (1.0 - true_labels)], dim=1))

# domain independent training requires change of the model outpu
class AdvLearningClassifier(Classifier):
    def __init__(self, input_shape, adversary_loss_weight=0.1, sensitive=None, device=None):
        super().__init__(input_shape=input_shape, sensitive=sensitive, device=device)
        self.criterion_bias = nn.CrossEntropyLoss(reduction='none')
        self.adv_weight = adversary_loss_weight

    def get_model(self, input_shape):
        return Net(input_shape, 1, 50) # domain independent classifier requires an prediction for each domain

    def _adversary_model(self, pred_logits, true_labels):
        c = torch.nn.Parameter(torch.Tensor([1.0]))
        s = torch.sigmoid((1 + torch.abs(c)) * pred_logits)
        adversary = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        return adversary(torch.cat([s, s * true_labels, s * (1.0 - true_labels)], dim=1))

    def loss(self, x, y, z, w=None):
        if w is None:
            w = torch.ones(x.shape[0]).to(self.device)
        y_hat, z_hat = self.model(x)
        y_hat, y = torch.squeeze(y_hat, dim=1), torch.squeeze(y, dim=1)
        loss = self.criterion(y_hat, y) + \
            self.criterion_bias(z_hat, z.long())
        return loss*w, y_hat

    def predict(self, x, z):
        self.model.eval()
        y_hat, z_hat = self.model(x)
        # y_hat = torch.unsqueeze(y_hat,dim=-1)
        return y_hat
