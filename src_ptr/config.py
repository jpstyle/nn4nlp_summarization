# train_file_path = "data/try_out/pubmed/train.orig.txt"
# val_file_path = "data/try_out/pubmed/val.txt"
# test_file_path = "data/try_out/pubmed/test.txt"
# vocab_file_path = "data/pubmed/vocab"
train_file_path = "data/pubmed/train.txt"
val_file_path = "data/pubmed/val.txt"
test_file_path = "data/pubmed/test.txt"
vocab_file_path = "data/pubmed/vocab"
log_root = "logs"

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 8
max_enc_steps=2000
max_dec_steps=210
beam_size=4
min_dec_steps=35
vocab_size=50006

log_interval=20
save_interval=100

max_section_len=500
max_num_sections=4
batch_shuffle_window=1
seed=11

ep=10000

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
