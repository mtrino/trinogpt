import os
import time
import math
import torch
import tiktoken
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.utils.dataloader import DataLoader
from src.utils.ddp import setup_ddp, cleanup_ddp
from src.utils.lr import cosine_lr_decay
from src.utils.generate import generate
from configs.pretrain import ModelConfig, TrainingConfig
from src.models.model import Model
from eval_scripts.hellaswag import iterate_examples, render_example

# DEVICE ---------------------------------------------------------------------------------------------------------------------------
ddp_env = setup_ddp()

# Setting the seeds for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

total_batch_size = TrainingConfig().total_batch_size
micro_batch_size = TrainingConfig().micro_batch_size
seq_len = ModelConfig().context_length

assert total_batch_size % (micro_batch_size * seq_len * ddp_env['ddp_world_size']) == 0, "make sure total_batch_size is divisible by (B * T * ddp_world_size)"
grad_accum_steps = total_batch_size // (micro_batch_size * seq_len * ddp_env['ddp_world_size'])
if ddp_env['master_process']:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(B=micro_batch_size, T=seq_len, process_rank=ddp_env['rank'], num_processes=ddp_env['ddp_world_size'], split="train")
val_loader = DataLoader(B=micro_batch_size, T=seq_len, process_rank=ddp_env['rank'], num_processes=ddp_env['ddp_world_size'], split="val")

# Using tensor cores using tf32 matrix multiply
torch.set_float32_matmul_precision('high') # enabling tf32

model = Model(ModelConfig())
model.to(ddp_env['device'])
if ModelConfig().use_compile:
    model = torch.compile(model)
if ddp_env['is_ddp']:
    model = DDP(model, device_ids=[ddp_env['ddp_local_rank']])
raw_model = model.module if ddp_env['ddp'] else model

max_lr = TrainingConfig().max_lr
min_lr = TrainingConfig().min_lr
warmup_steps = TrainingConfig().warmup_steps
max_steps = TrainingConfig().max_steps

optimizer = raw_model.configure_optimizers(weight_decay=TrainingConfig().weight_decay, learning_rate=max_lr)

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

enc = tiktoken.get_encoding("gpt2")

# create the log directory to write all checkpoints
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = (os.path.join(log_dir, f"log.txt"))
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    if step % TrainingConfig().val_every == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(ddp_env["device"]), y.to(ddp_env["device"])
                with torch.autocast(device_type=ddp_env["device"], dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= val_loss_steps
                val_loss_accum += loss.detach()
        if ddp_env["is_ddp"]:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if ddp_env["master_process"]:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % TrainingConfig().checkpoint_every == 0 or last_step):
                # write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    if (step % TrainingConfig().eval_every == 0 or last_step) and (not ModelConfig().use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_env["ddp_world_size"] != ddp_env["ddp_rank"]:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(ddp_env["device"])
            mask = mask.to(ddp_env["device"])
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=ddp_env["device"], dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp_env["is_ddp"]:
            num_total = torch.tensor(num_total, dtype=torch.long, device=ddp_env["device"])
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=ddp_env["device"])
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if ddp_env["master_process"]:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    if step > 0 and (step % TrainingConfig().generate_every == 0 or last_step):
        model.eval()
        num_return_sequences = TrainingConfig().num_return_sequences
        max_length = TrainingConfig().max_length
        generate(model, "Hello, I'm a language model", num_return_sequences, max_length, ddp_env)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(ddp_env["device"]), y.to(ddp_env["device"])
        with torch.autocast(device_type=ddp_env["device"], dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp_env["is_ddp"]:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # internal to pytorch no_sync() context manager
        loss.backward()
    if ddp_env["is_ddp"]:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = cosine_lr_decay(step, warmup_steps, max_lr, min_lr, max_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if ddp_env["device"].startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in milliseconds
    tokens_per_second = (train_loader.B * train_loader.T * grad_accum_steps * ddp_env["ddp_world_size"]) / ((t1 - t0))
    if ddp_env["master_process"]:
        print(f"step: {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:E} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second}")

cleanup_ddp(ddp_env["is_ddp"])