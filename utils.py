import torch

def get_distribution(logits, temperature):
    if temperature == 0:
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(
            -1, logits.argmax(dim=-1, keepdim=True), 1.0
        )
    return torch.softmax(logits.float() / temperature, dim=-1)

def sample(logits, temperature):
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)[0]
    probs = get_distribution(logits, temperature)
    return sample_probabilities(probs)[0]


def sample_probabilities(probs):
    # Exponential races sample unnormalized weights without a CPU-GPU validation sync.
    noise = torch.empty_like(probs).exponential_()
    return (probs / noise).argmax(dim=-1, keepdim=True)
