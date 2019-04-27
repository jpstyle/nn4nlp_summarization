# _nn4nlp_summarization

Make sure pytorch, tensorflow, pyrouge is installed. This code can be run on python3.6+, and any version of tensorflow; tensorflow is used only for logging.

Before running the code, mkdir `log`, `log/models`, `models`.

Place train.txt and test.txt in `data/pubmed` directory.

For training, specify GPUs to use, training mode, number of epochs & batch size. Unless specified, checkpoints will be saved at the end of every epoch. Provide `-save_interval` to additionally save checkpoints with some iteration interval. If continuing training from saved checkpoint, specify `-load_from`. If starting training with coverage, add `-cov`.

Training script example:

  ```
  $ python3 src/main.py -mode train -gpus 0 1 2 3 4 5 6 7 -ep 3 -batch_size 64 -cov -load_from models/some_exp/some_checkpoint
  ```

For decoding, only use one GPUs at most. Specify beam size with `-batch_size`. Choose model to test with `-load_from`. If the model checkpoint is using coverage, don't forget to provide `-cov`.

  ```
  $ python3 src/main.py -mode decode -gpus 0 -batch_size 8 -cov -load_from models/some_exp/some_checkpoint`
  ```
