import torch
import torch.nn as nn

from tqdm import tqdm

from .hardware_linear import HardwareLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_accuracy_bp(model, dataloader):
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            predicted = torch.max(outputs, 1).indices
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total


def compute_grads_bp(model, feats, labels, _):
    unnormalized_logits = model(feats)
    loss = nn.functional.cross_entropy(unnormalized_logits, labels)
    loss.backward()
    return loss


def compute_grads_hardware(model: HardwareLinear, feats, labels, _):
    unnormalized_logits = model(feats)
    loss = nn.functional.cross_entropy(unnormalized_logits, labels)

    # Custom backward
    logits = nn.functional.softmax(unnormalized_logits, dim=-1)
    ys = nn.functional.one_hot(labels, num_classes=4)
    model[0].weight.grad = (
        (logits - ys).unsqueeze(-1) @ model[0].saved_input.unsqueeze(-1).mT
    ).mean(dim=0)

    return loss


def _ff_gen_pos(feats: torch.Tensor, labels: torch.Tensor):
    pos_feats = torch.zeros((feats.size(0), feats.size(1) + 4), device=device)
    pos_feats[:, :4] = nn.functional.one_hot(labels, num_classes=4)
    pos_feats[:, 4:] = feats
    return pos_feats


def _ff_gen_neg(feats: torch.Tensor, labels: torch.Tensor):
    neg_feats = torch.zeros((feats.size(0), feats.size(1) + 4), device=device)
    neg_labels = torch.randint(4, (feats.size(0),), device=device)
    neg_labels = torch.where(
        labels == neg_labels, (neg_labels + 1) % 4, neg_labels
    )  # Change if pos label
    neg_feats[:, :4] = nn.functional.one_hot(neg_labels, num_classes=4)
    neg_feats[:, 4:] = feats
    return neg_feats


def _ff_gen_neutral(feats: torch.Tensor):
    neutral_feats = torch.zeros((feats.size(0), feats.size(1) + 4), device=device)
    neutral_feats[:, :4] = 1 / 4
    neutral_feats[:, 4:] = feats
    return neutral_feats


def compute_accuracy_ff(model, dataloader, *, for_layer=-1):
    model.eval()

    if for_layer < 0:
        for_layer = len(model) + for_layer

    neurons_per_cluster = model[-1][0].out_features // 4
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            clusters_activity = torch.zeros((inputs.size(0), 4), device=device)
            inputs, labels = inputs.to(device), labels.to(device)

            acts = (
                _ff_gen_neutral(inputs)
                if model[0][0].in_features != inputs.size(1)
                else inputs
            )
            for layer in model[: for_layer + 1]:
                acts = layer(acts)

            for cluster_idx in range(4):
                acts_cluster = acts[
                    :,
                    cluster_idx * neurons_per_cluster : (cluster_idx + 1)
                    * neurons_per_cluster,
                ]
                clusters_activity[:, cluster_idx] += acts_cluster.sum(dim=1)

            correct += (clusters_activity.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total


def compute_grads_ff(model, feats, labels, thetass):
    losses = []
    acts_pos = _ff_gen_pos(feats, labels)
    acts_neg = _ff_gen_neg(feats, labels)

    # To simplify, just compute the grads of all layers instead of one by one
    # The one(s) not concerned will be zeroed out later
    # (Overhead negligible wrt time spent to program & read memristors)
    for layer, thetas in zip(model.children(), thetass):
        acts_pos = layer(acts_pos)
        acts_neg = layer(acts_neg)
        goodness_pos = acts_pos.square().sum(dim=1)
        goodness_neg = acts_neg.square().sum(dim=1)

        loss = -nn.functional.logsigmoid(
            torch.cat(
                [
                    +(goodness_pos - thetas[0] * acts_pos.size(1)),
                    -(goodness_neg - thetas[1] * acts_neg.size(1)),
                ]
            )
        ).mean()
        loss.backward()
        losses.append(loss)

        # Detach from graph to start over independant grad computation next layer
        acts_pos.detach_()
        acts_neg.detach_()

    return acts_pos, sum(losses)


def compute_grads_clusters_hidden(model, feats, labels, thetass):
    losses = []
    acts = feats

    # To simplify, just compute the grads of all layers instead of one by one
    # The one(s) not concerned will be zeroed out later
    # (Overhead negligible wrt time spent to program & read memristors)
    for layer, thetas in zip(model.children(), thetass):
        num_neurons = layer[0].out_features
        neurons_per_cluster = num_neurons // 4
        labels_oh = nn.functional.one_hot(labels, num_classes=4)
        cluster_mask = torch.kron(labels_oh, torch.ones((1, neurons_per_cluster)))

        # No pos or neg data anymore, only pos or neg clusters, def. by cluster_mask
        acts = layer(acts)
        goodness_pos = ((1 - cluster_mask) * acts).square().sum(dim=1)
        goodness_neg = (cluster_mask * acts).square().sum(dim=1)

        loss = -nn.functional.logsigmoid(
            torch.cat(
                [
                    +(goodness_pos * thetas[0]),
                    -(goodness_neg * thetas[1]),
                ]
            )
        ).mean()
        loss.backward()
        losses.append(loss)

        # Detach from graph to start over independant grad computation next layer
        acts.detach_()

    return acts, sum(losses)


def compute_grads_clusters_classifier(model, feats, labels, thetass):
    losses = []
    acts = feats

    # To simplify, just compute the grads of all layers instead of one by one
    # The one(s) not concerned will be zeroed out later
    # (Overhead negligible wrt time spent to program & read memristors)
    for layer, thetas in zip(model.children(), thetass):
        num_neurons = layer[0].out_features
        neurons_per_cluster = num_neurons // 4
        labels_oh = nn.functional.one_hot(labels, num_classes=4)
        cluster_mask = torch.kron(labels_oh, torch.ones((1, neurons_per_cluster)))

        # No pos or neg data anymore, only pos or neg clusters, def. by cluster_mask
        acts = layer(acts)
        goodness_pos = (cluster_mask * acts).square().sum(dim=1)
        goodness_neg = ((1.0 - cluster_mask) * acts).square().sum(dim=1)

        loss = -nn.functional.logsigmoid(
            torch.cat(
                [
                    +(goodness_pos * thetas[0]),
                    -(goodness_neg * thetas[1]),
                ]
            )
        ).mean()
        loss.backward()
        losses.append(loss)

        # Detach from graph to start over independant grad computation next layer
        acts.detach_()

    return sum(losses)


def compute_grads_sff(model, feats, labels, thetas):
    # Unlike traditional Forward-Forward, we use Clusters for the last layer, thus slightly different learning rule
    acts, loss = compute_grads_ff(model[:-1], feats, labels, thetas[:-1])
    loss += compute_grads_clusters_classifier(model[-1:], acts, labels, thetas[-1:])

    return loss


def compute_grads_cf(model, feats, labels, thetas):
    acts, loss = compute_grads_clusters_hidden(model[:-1], feats, labels, thetas[:-1])
    loss += compute_grads_clusters_classifier(model[-1:], acts, labels, thetas[-1:])

    return loss