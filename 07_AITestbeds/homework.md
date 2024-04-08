## Sambanova

The ntasks flag specifies how many instances of the command to be executed. (number of tasks in a job or job step) 

When it is set to 8, the job takes slightly longer. When it is set to 32, the job fails due to invalid configuration. This makes sense, because we only requested 1 node in the command, with ntasks-per-node = 16 (--ntasks-per-node 16  --nodes 1)

## Graphcore

The script must be submitted to slurm, otherwise, they fail due to unavaialbility of IPUs, with this error:

```
[09:01:59.258] [poptorch::python] [critical] poptorch.poptorch_core.Error: In poptorch/popart_compiler/source/CompilerImpl.cpp:725: 'poptorch_cpp_error': Failed to acquire 1 IPU(s)                         
No IPU detected in the system. 
For more information use the Graphcore command-line tool `gc-monitor`.
Error raised in:
  [0] Compiler::initSession
  [1] LowerToPopart::compile
  [2] compileWithManualTracing


Traceback (most recent call last):
  File "mnist_poptorch.py", line 279, in <module>
    preds, losses = training_model(data, labels)
  File "/home/aeahmed/venvs/graphcore/poptorch33_env/lib/python3.8/site-packages/poptorch/_poplar_executor.py", line 1238, in __call__
    self._compile(in_tensors)
  File "/home/aeahmed/venvs/graphcore/poptorch33_env/lib/python3.8/site-packages/poptorch/_impl.py", line 358, in wrapper
    return func(self, *args, **kwargs)
  File "/home/aeahmed/venvs/graphcore/poptorch33_env/lib/python3.8/site-packages/poptorch/_poplar_executor.py", line 975, in _compile
    self._executable = self._compileWithDispatch(in_tensors_trace_view)
  File "/home/aeahmed/venvs/graphcore/poptorch33_env/lib/python3.8/site-packages/poptorch/_impl.py", line 164, in wrapper
    return func(*args, **kwargs)
  File "/home/aeahmed/venvs/graphcore/poptorch33_env/lib/python3.8/site-packages/poptorch/_poplar_executor.py", line 936, in _compileWithDispatch
    executable = poptorch_core.compileWithManualTracing(
poptorch.poptorch_core.Error: In poptorch/popart_compiler/source/CompilerImpl.cpp:725: 'poptorch_cpp_error': Failed to acquire 1 IPU(s)
No IPU detected in the system. 
For more information use the Graphcore command-line tool `gc-monitor`.
Error raised in:
  [0] Compiler::initSession
  [1] LowerToPopart::compile
  [2] compileWithManualTracing

```
It is strange that running with small number of epochs yielded the highest accuracy. The logs below are for runs with 10, 4, and 14 epochs, which yielded test accuracies of 97.13%, 98.58%, and 98.43%, respectively. 


### When run with default hyperparameters:

learning_rate = 0.03

epochs = 10

batch_size = 8

test_batch_size = 80


```
(poptorch33_env) aeahmed@gc-poplar-02:~/graphcore/examples/tutorials/simple_applications/pytorch/mnist$ /opt/slurm/bin/srun --ipus=1 python mnist_poptorch.py
srun: job 20702 queued and waiting for resources
srun: job 20702 has been allocated resources
Epochs:   0%|          | 0/10 [00:00<?,[09:03:32.515] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 10
  0%|          | 0/150 [00:00<?, ?it/s]                2024-04-08T09:03:33.195878Z PL:POPLIN    773471.773471 W: poplin::preplanConvolution() is deprecated! Use poplin::preplan() instead
                                                       2024-04-08T09:03:36.656904Z PL:POPLIN    773471.773471 W: poplin::preplanMatMuls() is deprecated! Use poplin::preplan() instead
Graph compilation: 100%|██████████| 100/100 [00:22<00:00]2024-04-08T09:03:55.437249Z popart:session 773471.773471 W: Rng state buffer was not serialized.You did not load poplar Engine.Remember that if you would like to run the model using the model runtime then you have to create your own buffer and callback in your model runtime application for rngStateTensor.
Graph compilation:  98%|█████████▊| 98/100 [00:22<00:01]
Epochs: 100%|██████████| 10/10 [01:49<00:00, 10.99s/it]
  0%|          | 0/125 [00:00<?, ?it/s]                2024-04-08T09:05:22.914100Z PL:POPLIN    773471.773471 W: poplin::preplanConvolution() is deprecated! Use poplin::preplan() instead
Graph compilation:   4%|▍         | 4/100 [00:00<00:04]2024-04-08T09:05:25.096615Z PL:POPLIN    773471.773471 W: poplin::preplanMatMuls() is deprecated! Use poplin::preplan() instead
Graph compilation: 100%|██████████| 100/100 [00:14<00:00]
 97%|██████TrainingModelWithLoss(0:00, 13.97it/s]<00:00]
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 97.13%

```

### And with a smaller number of epochs:

learning_rate = 0.03

epochs = 4

batch_size = 8

test_batch_size = 80


```
(poptorch33_env) aeahmed@gc-poplar-02:~/graphcore/examples/tutorials/simple_applications/pytorch/mnist$ /opt/slurm/bin/srun --ipus=1 python mnist_poptorch_smaller_epochs.py 
srun: job 20703 queued and waiting for resources
srun: job 20703 has been allocated resources
Epochs:   0%|          | 0/10 [00:00<?,[09:07:03.089] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 10
Graph compilation: 100%|██████████| 100/100 [00:00<00:00]
Epochs: 100%|██████████| 10/10 [01:27<00:00,  8.79s/it]
Graph compilation: 100%|██████████| 100/100 [00:00<00:00]                          
 95%|███████TrainingModelWithLoss(:00, 45.51it/s]0<00:00]
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 98.58%
```

### And with larger epochs:

learning_rate = 0.03

epochs = 14

batch_size = 8

test_batch_size = 80


```
(poptorch33_env) aeahmed@gc-poplar-02:~/graphcore/examples/tutorials/simple_applications/pytorch/mnist$ /opt/slurm/bin/srun --ipus=1 python mnist_poptorch_larger_epochs.py 
srun: job 20704 queued and waiting for resources
srun: job 20704 has been allocated resources
Epochs:   0%|          | 0/10 [00:00<?,[09:09:34.130] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 10
Graph compilation: 100%|██████████| 100/100 [00:00<00:00]
Epochs: 100%|██████████| 10/10 [01:27<00:00,  8.74s/it]
Graph compilation: 100%|██████████| 100/100 [00:00<00:00]                          
 86%|████████▌ | 107/125 [00:TrainingModelWithLoss(00:00]
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 98.43%
```

### Note

It would be useful to indicate that the proper way to call the profiler is as follows:

```
(poptorch33_env) aeahmed@gc-poplar-02:~/graphcore/examples/tutorials/simple_applications/pytorch/mnist$ /opt/slurm/bin/srun --ipus=1  POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"/home/aeahmed/graph_profile", "profiler.includeFlopEstimates": "true"}' python mnist_poptorch.py
srun: job 20707 queued and waiting for resources
srun: job 20707 has been allocated resources
slurmstepd: error: execve(): POPLAR_ENGINE_OPTIONS={"autoReport.all":"true", "autoReport.directory":"/home/aeahmed/graph_profile", "profiler.includeFlopEstimates": "true"}: No such file or directory
srun: error: gc-poplar-03: task 0: Exited with exit code 2
(poptorch33_env) aeahmed@gc-poplar-02:~/graphcore/examples/tutorials/simple_applications/pytorch/mnist$ POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"/home/aeahmed/graph_profile", "profiler.includeFlopEstimates": "true"}' /opt/slurm/bin/srun --ipus=1  python mnist_poptorch.py
srun: job 20708 queued and waiting for resources
srun: job 20708 has been allocated resources
Epochs:   0%|          | 0/10 [00:00<?,[09:24:38.110] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 10
  0%|          | 0/150 [00:00<?, ?it/s]                2024-04-08T09:24:38.561616Z PL:POPLIN    791072.791072 W: poplin::preplanConvolution() is deprecated! Use poplin::preplan() instead
                                                       2024-04-08T09:24:42.037132Z PL:POPLIN    791072.791072 W: poplin::preplanMatMuls() is deprecated! Use poplin::preplan() instead
Graph compilation: 100%|██████████| 100/100 [00:28<00:00]2024-04-08T09:25:07.924981Z popart:session 791072.791072 W: Rng state buffer was not serialized.You did not load poplar Engine.Remember that if you would like to run the model using the model runtime then you have to create your own buffer and callback in your model runtime application for rngStateTensor.
Graph compilation:  98%|█████████▊| 98/100 [00:28<00:03]
Epochs: 100%|██████████| 10/10 [02:16<00:00, 13.63s/it]
  0%|          | 0/125 [00:00<?, ?it/s]                2024-04-08T09:26:54.821320Z PL:POPLIN    791072.791072 W: poplin::preplanConvolution() is deprecated! Use poplin::preplan() instead
Graph compilation:   4%|▍         | 4/100 [00:00<00:03]2024-04-08T09:26:57.016313Z PL:POPLIN    791072.791072 W: poplin::preplanMatMuls() is deprecated! Use poplin::preplan() instead
Graph compilation:   6%|▌         | 6/100 [00:02<00:33] 2024-04-08T09:27:07.604347Z PO:ENGINE   791072.791072 W: Estimated counters: 188416, actual counters: 59971
Graph compilation: 100%|██████████| 100/100 [00:19<00:00]
 90%|█████████ | 113/TrainingModelWithLoss(3it/s]<00:04]
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 98.13%

```

## Groq

### Default run:

```
Info: No inputs received for benchmark. Using the inputs provided during model compilation.
Running inference on GroqChip.
Running inference using PyTorch model (CPU).
100%|███████████████████████████████████████████████████████████████████████| 2210/2210 [00:04<00:00, 453.50it/s]
+--------+----------+-------------------------+----------------+----------------------+-------------+
| Source | Accuracy | end-to-end latency (ms) | end-to-end IPS | on-chip latency (ms) | on-chip IPS |
+--------+----------+-------------------------+----------------+----------------------+-------------+
|  cpu   |  77.47%  |           2.21          |     453.45     |          --          |      --     |
|  groq  |  77.47%  |           0.05          |    19253.91    |         0.03         |   37576.72  |
+--------+----------+-------------------------+----------------+----------------------+-------------+
Proof point /home/aeahmed/groqflow/proof_points/natural_language_processing/bert/bert_tiny.py finished!

```



## Cerebras


### Default run:

```
Processed 1024000 sample(s) in 210.762417634 seconds.
```

### with 512 batch size

```
2024-04-08 16:38:14,888 INFO:   Processed 512000 sample(s) in 175.796717907 seconds.
```


