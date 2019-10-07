###### Tensorflow 的架构
Tensorflow 架构包括：C++， Python的前端接口； Core Tensorflow Execution System 中间层； CPU, GPU, Android, IOS 的底层实现.

###### Setup Tensorflow
2 classical environment needed: Jupyter(ipython) Notebook, matplotlib.

###### setup in Virtulenv
    sudo easy_install pip 
    sudo pip install --upgrade virtualenv

###### install tensorflow
1. 2.7: pip install --upgrade tensorflow
2. 3.4: pip3 install --upgrade tensorflow
3. download whl file and install by 'pip install'

###### test
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inplne a
    a=tf.random_normal([2,20])
    sess=tf.Session()
    out=sess.run(a)
    x,y=out
    plt.scatter(x,y)
    plt.show()

###### Tensorflow工作流图
节点 Operation
    import tensorflow as tf
    a=tf.constant(5,name="input_a") #define the operation a, create the constant operation , output t connected operations
    b=tf.constant(3,name="input_b") #define the operation b, create the constant operation, output to connected operations
    c=tf.multiply(a,b,name="mul_c") #define the operation c, create the multiply operation, output the product
    d=tf.add(a,b,name="add_d") #define the operation d, create the add operation, output the sum
    e=tf.add(c,d,name="add_e") #define the operation e, create the add operation, output the sum
    sess=tf.Session()#create and start the object of Tensorflow Session
    output=sess.run(e) #Session object carries out the operation e and save the output result
    writer=tf.summary.FileWriter('./my_graph',sess.graph) #create and start the object of Summary.FileWriter, 2 parameters needed: tensorboard image save path; property of sess.graph
    writer.close() #shut down the summary.FileWriter object
    sess.close() #shut down the Session object
