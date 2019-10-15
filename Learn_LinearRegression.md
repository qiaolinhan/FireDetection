##### Linear Regression
* supervised learnig, input the labeled dataset.
* train the closed loop

`initialize the model parameters`-->`input detaset (shuffled)`-->`train he datasets and get the output`
-->`compute the cost`-->`adjust the parameters of the model`

* set the checking point when every 1000 training step ends or the whole training step ends.
checking point file is create by name `my-model{step}`
```python
tf.train.Saver.save
```
Gradient Decent is used to optimize the parameters of the model. To minimize the value of cost function.
```python
import tensorflow as tf
import os
#initialize the variables and model parameters, difine the closed loop training
W=tf.Variable(tf.zeros([2,1],name="weights")) #variable weights
b=tf.Variable(0.,name="bias") #the bias of model
def inference(X):  #for the data X, compute result return to the output of this inferred model
  print ("function: inference")
  return tf.matmul(X,W)+b
def loss(X,Y): #compute the loss, for the data X and expected output Y
      print "function: loss"
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))
def inputs():#读取或生成训练数据X及期望输出Y
    print "function: inputs"
    # Data is from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)
def train(total_loss):#训练或调整模型参数(计算总损失)
    print "function: train"
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
def evaluate(sess, X, Y):#评估训练模型
    print "function: evaluate"
    print sess.run(inference([[80., 25.]])) # ~ 303
    print sess.run(inference([[65., 25.]])) # ~ 256
saver = tf.train.Saver()#创建Saver对象
#会话对象启动数据流图，搭建流程
with tf.Session() as sess:
    print "Session: start"
    tf.global_variables_initializer().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    training_steps = 1000#实际训练迭代次数
    initial_step = 0
    checkpoin_dir = "./"
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoin_dir))
    if ckpt and ckpt.model_checkpoint_path:
        print "checkpoint_path: " + ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)#从检查点恢复模型参数
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
    for step in range(initial_step, training_steps):#实际训练闭环
        sess.run([train_op])
    if step % 10 == 0:#查看训练过程损失递减
        print str(step)+ " loss: ", sess.run([total_loss])
        save_file = saver.save(sess, 'my-model', global_step=step)#创建遵循命名模板my-model-{step}检查点文件
        print str(step) + " save_file: ", save_file
    evaluate(sess, X, Y)#模型评估
    coord.request_stop()
    coord.join(threads)
    saver.save(sess, 'my-model', global_step=training_steps)
    print str(training_steps) + " final loss: ", sess.run([total_loss])
    sess.close()
```

##### Sigmoid
```python
import tensorflow as tf
import os
#参数变量初始化
W = tf.Variable(tf.zeros([5, 1]), name="weights")#变量权值
b = tf.Variable(0., name="bias")#线性函数常量，模型偏置
def combine_inputs(X):#输入值合并
    print "function: combine_inputs"
    return tf.matmul(X, W) + b
def inference(X):#计算返回推断模型输出(数据X)
    print "function: inference"
    return tf.sigmoid(combine_inputs(X))#调用概率分布函数
def loss(X, Y):#计算损失(训练数据X及期望输出Y)
    print "function: loss"
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))#求平均值
def read_csv(batch_size, file_name, record_defaults):#从csv文件读取数据，加载解析，创建批次读取张量多行数据
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=record_defaults)#字符串(文本行)转换到指定默认值张量列元组，为每列设置数据类型
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)#读取文件，加载张量batch_size行
def inputs():#读取或生成训练数据X及期望输出Y
    print "function: inputs"
    #数据来源：https://www.kaggle.com/c/titanic/data
    #模型依据乘客年龄、性别、船票等级推断是否能够幸存
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])
    #转换属性数据
    is_first_class = tf.to_float(tf.equal(pclass, [1]))#一等票
    is_second_class = tf.to_float(tf.equal(pclass, [2]))#二等票
    is_third_class = tf.to_float(tf.equal(pclass, [3]))#三等票
    gender = tf.to_float(tf.equal(sex, ["female"]))#性别，男性为0，女性为1

    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))#所有特征排列矩阵，矩阵转置，每行一样本，每列一特征
    survived = tf.reshape(survived, [100, 1])
    return features, survived
def train(total_loss):#训练或调整模型参数(计算总损失)
    print "function: train"
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
def evaluate(sess, X, Y):#评估训练模型
    print "function: evaluate"
    predicted = tf.cast(inference(X) > 0.5, tf.float32)#样本输出大于0.5转换为正回答
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))#统计所有正确预测样本数，除以批次样本总数，得到正确预测百分比
#会话对象启动数据流图，搭建流程
with tf.Session() as sess:
    print "Session: start"
    tf.global_variables_initializer().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    training_steps = 1000#实际训练迭代次数
    for step in range(training_steps):#实际训练闭环
        sess.run([train_op])
        if step % 10 == 0:#查看训练过程损失递减
            print str(step)+ " loss: ", sess.run([total_loss])
    print str(training_steps) + " final loss: ", sess.run([total_loss])
    evaluate(sess, X, Y)#模型评估
    import time
    time.sleep(5)
    coord.request_stop()
    coord.join(threads)
    sess.close()
 
