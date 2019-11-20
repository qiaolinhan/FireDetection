#### Tensor representation and classes, shapes, calculation
The input opertion turned from receiving scalars into receiving vectors
* .ruduce_prod() create the product operation
* .reduce_sum() create the sum operation  
```python
import tensorflow as tf
a = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[1,2,3]]], name="input_a")
b = tf.reduce_prod(a, name="prod_b")#定义节点b，创建归约乘积Op,接收一个N维张量输入,输出张量所有分量(元素)的乘积,以prod_b标识。
c = tf.reduce_sum(a, name="sum_c")#定义节点c，创建归约求和Op,接接收一个N维张量输入,输出张量所有分量(元素)的求和,以sum_c标识。
d = tf.add(b,c, name="add_d")
sess = tf.Session()
out = sess.run(d)
writer = tf.summary.FileWriter('./my_graph', sess.graph)
writer.close()
sess.close()
```
Data in tensorflow is based on Numpy, any Numpy data can transfer to tensorflow operation.  
There is no exact data class to `Tensorflow.string`.
It is better to point the tensor OP by Numpy `BY HAND`
The shape of tensor contains `list` and `tuple`
```python
import tensorflow as tf
#指定0阶张量(标量)形状
s_0 = 1
#指定1阶张量(向量)形状
s_1_list = [1,2,3]
s_1_tuple = (1,2,3)
#指定2阶张量(矩阵)形状
s_2 = [[2,3],[2,3]]
#指定任意维数任意长度的张量形状
s_any = None
shape = tf.shape(s_2, name="mystery_shape")#创建获取张量形状Op,接收一个张量，输出张量形状,以mystery_shape标识。
sess = tf.Session()
sess.run(shape)
```
###### Calculation
`-x，.neg()`，x中每个元素的相反数。
`~x，.logical_not()`，x中每个元素的逻辑非，只适用dtype为tf.bool的Tensor对象。
`abc()，.abs()`，x中每个元素的绝对值。
`x+y，.add()`，x、y逐元素相加。
`x-y，.sub()`，x、y逐元素相减。
`x*y，.multiply()`，x、y逐元素相乘。
`x/y，.div()`，x、y逐元素相除，整数张量执行整数除法，浮点数张量执行浮点数除法。
`x%y，.mod()`，逐元素取模。
`x**y，.pow()`，x逐元素为底，y逐元素为指数的幂。
`x<y，.less()`，逐元素计算x<y真值。
`x<=y，.less_equal()`，逐元素计算x<=y真值。
`x>y，.greater()`，逐元素计算x>y真值。
`x>=y，.greater_equal()`，逐元素计算x>=y真值。
`x&y，.logical_and()`，逐元素计算x&y真值，元素dtype必须为tf.bool。
`x|y，.logical_or()`，逐元素计算x|y真值，元素dtype必须为tf.bool。
`x^y，.logical_xor()`，逐元素计算x^y真值，元素dtype必须为tf.bool。 
运算符重载无法为Op指定name。==判断两个Tensor对象名是否引用同一个对象。.equal()和.not_equal()判断张量值是否相同。
