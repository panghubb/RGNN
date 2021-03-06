import copy

# data
data_path = './data'
vocab_path = './data/essay.vocab'

# mode = 'train'  # must be one of train/decode
mode = 'decode'

# log
log_root = 'log'
# exp_name = 'test-new-generate-drop-init'
exp_name = 'test-standard-attdec'
max_to_keep = 40
model_save_freq = 5
sample_path = 'result.txt'

emb_dim = 300
hidden_dim = 800
num_layers = 2
max_enc_steps = 120
num_keywords = 5

keep_prob = 0.5

vocab_size = 50000
max_epochs = 41
batch_size = 16
beam_size = 2
max_gen_steps = 160

learning_rate = 0.001
# learning_rate = 0.0001
rand_unif_init = 0.02
max_grad_norm = 5.0

train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "allow_smaller_final_batch": False,
    "seed": 123,
    "datasets": [
        {
            "files": './data/essay.train',
            "vocab_file": vocab_path,
            "max_seq_length": max_enc_steps,
            # "pad_to_max_seq_length": True,
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "data_name": "x",
            # "embedding_init": {
            #     "file": './data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5',
            #     "dim": emb_dim,
            #     "read_fn": "load_glove",
            #     "init_fn": {
            #         "type": "numpy.random.uniform",
            #         "kwargs": {
            #             "low": -rand_unif_init,
            #             "high": rand_unif_init,
            #         }
            #     },
            # },
        },
        {
            "files": './data/essay.train.label',
            "vocab_file": vocab_path,
            "max_seq_length": num_keywords,
            "bos_token": "",
            "eos_token": "",
            "data_name": "y",
        },
        {
            "files": './data/essay.train.top2',
            "vocab_file": vocab_path,
            "max_seq_length": max_enc_steps - 40,
            # "pad_to_max_seq_length": True,
            "bos_token": "",
            "eos_token": "",
            "data_name": "z",
        }
    ]
}

val_data_hparams = copy.deepcopy(train_data_hparams)
val_data_hparams["datasets"][0]["files"] = './data/essay.dev'
val_data_hparams["datasets"][1]["files"] = './data/essay.dev.label'
val_data_hparams["datasets"][2]["files"] = './data/essay.dev.top2'

test_data_hparams = copy.deepcopy(train_data_hparams)
test_data_hparams["datasets"][0]["files"] = './data/essay.test'
test_data_hparams["datasets"][1]["files"] = './data/essay.test.label'
test_data_hparams["datasets"][2]["files"] = './data/essay.test.top2'

embedder = {
    'dim': emb_dim,
}

encoder = {
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': 800
        }
    }
}

opt = {
    'optimizer': {
        'type':  'AdamOptimizer',
        'kwargs': {
            'learning_rate': learning_rate,
        },
    },
    'gradient_clip': {
        'type': 'clip_by_global_norm',
        'kwargs': {'clip_norm': max_grad_norm}
    },
}