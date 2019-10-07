##### Linear Regression
* supervised learnig, input the labeled dataset.
* train the closed loop

`initialize the model parameters`-->`input detaset (shuffled)`-->`train he datasets and get the output`
-->`compute the cost`-->`adjust the parameters of the model`

* set the checking point when every 1000 training step ends or the whole training step ends.
checking point file is create by name `my-model{step}`
```python
tf.train.Server.save
```
