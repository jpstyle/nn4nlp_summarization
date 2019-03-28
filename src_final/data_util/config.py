import os

#root_dir = os.path.expanduser("pointer_summarizer")
root_dir = os.path.expanduser("./")

# train_data_path = os.path.join(root_dir, "finished_files/chunked/train_*")
# eval_data_path = os.path.join(root_dir, "finished_files/val.bin")
# decode_data_path = os.path.join(root_dir, "finished_files/test.bin")
# vocab_path = os.path.join(root_dir, "finished_files/vocab")
train_data_path = os.path.join(root_dir, "pubmed_one/train.bin")
eval_data_path = os.path.join(root_dir, "pubmed/val.bin")
decode_data_path = os.path.join(root_dir, "pubmed/test.bin")
vocab_path = os.path.join(root_dir, "pubmed/vocab")
log_root = os.path.join(root_dir, "log")

save_dir="test"
print_interval=30
save_interval=3000

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 1

max_num_sec = 4
max_sec_len = 500
max_enc_steps=2000

max_dec_steps=210
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15

