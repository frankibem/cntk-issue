Repository to duplicate the issue [here](https://github.com/Microsoft/CNTK/issues/2505). In summary, on occasion either:
- A very large or very small positive value is reported for the evaluation metric: classification error
- The minibatch level statistics are in their correct ranges (loss < 50, 0 <= eval <= 100%) but the summary statistics are reported as being very large or very small

Dataset links:
* Training set: https://frankstoredb.blob.core.windows.net/asllvd/project/20/train.cbf
* Test set: https://frankstoredb.blob.core.windows.net/asllvd/project/20/test.cbf

Assuming the dataset has been downloaded to ~/dataset, you can run the code with
```
python train.py -if ~/dataset
```

- Logs will be redirected to ./logs/log.txt unless another output directory is specified.
- I was running CNTK v2.2 on a virtual machine with Ubuntu 16.04 (4 vCPUs, 16GB, 1 K80 Tesla GPU)
