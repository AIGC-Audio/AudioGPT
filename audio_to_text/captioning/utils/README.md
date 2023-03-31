# Utils

Scripts in this directory are used as utility functions.

## BERT Pretrained Embeddings

You can load pretrained word embeddings in Google [BERT](https://github.com/google-research/bert#pre-trained-models) instead of training word embeddings from scratch. The scripts in `utils/bert` need a BERT server in the background. We use BERT server from [bert-as-service](https://github.com/hanxiao/bert-as-service).

To use bert-as-service, you need to first install the repository. It is recommended that you create a new environment with Tensorflow 1.3 to run BERT server since it is incompatible with Tensorflow 2.x.

After successful installation of [bert-as-service](https://github.com/hanxiao/bert-as-service), downloading and running the BERT server needs to execute:

```bash
bash scripts/prepare_bert_server.sh <path-to-server> <num-workers> zh
```

By default, server based on BERT base Chinese model is running in the background. You can change to other models by changing corresponding model name and path in `scripts/prepare_bert_server.sh`.

To extract BERT word embeddings, you need to execute `utils/bert/create_word_embedding.py`.
