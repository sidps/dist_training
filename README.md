## Description

This repo contains all code related to my Masters thesis
involving distributed training of neural networks.

Link to Thesis on arXiv: https://arxiv.org/abs/1812.02407

----

## Installation:

Using virtualenv is highly recommend.
Developed using `Python 3.6.5` and `PyTorch 0.3.1`

Assuming you're using `Python 3.6.5` inside a virtual environment
with `pip` available, you will first need to install PyTorch

On a mac (no CUDA), use:

```bash
$ pip install http://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl
```

On Linux (with CUDA 8), use:

```bash
$ pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
```

(or pick a different binary with pytorch==0.3.1:
 https://pytorch.org/previous-versions/)

And then ...

```bash
$ pip install -r requirements.txt
```

----

## Execution

```bash
$ python main.py
```
will run the default "experiment" - the Iris classification task, 
with default configs - 4 workers, data split evenly, 
a 3-layer neural network with 3-way Softmax classifier, run
over 3 epochs using all-reduce.

This is tiny enough that it should run on any modern computer in
seconds, and serves well as a `Hello World`.

Run the following for more options:

```bash
$ python main.py --help
```

The most useful arguments would be:

```bash
--experiment {iris,mnist,cifar10}
--agg-method {local,noComm,gradAllReduce,elasticGossip,gossipingSgd}
                      aggregation method used to aggregate gradients or
                      params across all workers during training
--agg-period AGG_PERIOD
                      if applicable, the period at which an aggregation
                      occurs
--agg-prob AGG_PROB   if applicable, the probability with which agg occurs
--elastic-alpha ELASTIC_ALPHA
                      "moving rate" for elastic gossip
```

### Logging and output

Logs are Bunyan formatted, so you will need the Bunyan CLI tool 
to view them.

```bash
$ npm install bunyan
```

Logs are stored at `./logs/<exp-id>` where `<exp-id>` can be specified 
using the `--exp-id` argument, this defaults to `./logs/unspecified/`.

Logs are Bunynan-formatted, which means they're also JSON formatted. 
If you'd simply like to read them:

```bash
$ cat <logs> | bunyan -o short -l INFO
```

The logs folder has one log file for each worker, identified by rank, 
and a `metadata.json`, which is a dump of the command-line
arguments including the defaults.

```bash
$ cat ./logs/unspecified/metadata.json | jq
```

----

### Tests

```bash
$ python -m pytest
```
