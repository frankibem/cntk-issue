See [log.txt](https://github.com/frankibem/cntk-issue/blob/master/log.txt) for an example.

Repository contains necessary files to duplicate the issue [here](https://github.com/Microsoft/CNTK/issues/2505). In summary, on occasion either:
- A very large positive or negative value is reported for the summary (per epoch) statistics: loss and metric
- The minibatch level statistics are in their correct ranges (0 < loss, 0 <= eval <= 100%) 

Dataset links:
* Training set: https://frankstoredb.blob.core.windows.net/boston50/all/test0.cbf
* Test set: https://frankstoredb.blob.core.windows.net/boston50/all/train0.cbf

Assuming the dataset has been downloaded to ~/dataset, you can run the code with
```
python train.py -if ~/dataset -p "0"
```

- Logs will be redirected to ./logs/log.txt unless another output directory is specified.
- log.txt was generated on Ubuntu 16.04, CNTK v2.3.1, CPU-ONLY build (ran to completion)
- On windows 10 (CNTK v2.3.1, CPU-ONLY build) training terminates prematurely due to a crash with no error message.
- On Ubuntu 16.04 (CNTK v2.3.1, 1 K80 Tesla GPU), it sometimes runs to completion. Other times I get an 'illegal memory access' error as per this [issue](https://github.com/Microsoft/CNTK/issues/2691)
