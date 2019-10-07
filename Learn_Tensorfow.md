###### Tensorflow 的架构
Tensorflow 架构包括：C++， Python的前端接口； Core Tensorflow Execution System 中间层； CPU, GPU, Android, IOS 的底层实现.

###### Setup Tensorflow
2 classical environment needed: Jupyter(ipython) Notebook, matplotlib.

###### setup in Virtulenv
    sudo easy_install pip sudo pip install --upgrade virtualenv

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
