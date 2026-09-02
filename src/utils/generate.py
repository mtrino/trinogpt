import torch
import tiktoken
from torch.nn import functional as F

enc = tiktoken.get_encoding("gpt2")

def generate(model, prompt, num_sequences, max_length, ddp_env, k=50):
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_sequences, 1)
    xgen = tokens.to(ddp_env['device'])
    sample_rng = torch.Generator(device=ddp_env["device"])
    sample_rng.manual_seed(42 + ddp_env["rank"])
    while xgen.size(1) <= max_length:
        with torch.no_grad():
            logits = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=k)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat([xgen, xcol], dim=-1)
    for i in range(num_sequences):
        token_list = xgen[i, :max_length].tolist()
        decoded = enc.decode(token_list)
        print(f"rank {ddp_env["ddp_rank"]}, sample {i}: {decoded}")



