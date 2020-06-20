import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
def get_weight_variable(shape,name):
    initial=tf.contrib.layers.xavier_initializer( uniform=True, seed=None,
                                                dtype=tf.float32)
    return tf.get_variable(name,shape,initializer=initial)

def get_bias_variable(shape,name):
    initial=tf.zeros_initializer()
    return tf.get_variable(name,shape,initializer=initial)

def conv2d(x,W,strides=(1,1,1,1)):
    return tf.nn.conv2d(x,W,strides=strides,padding="SAME")

def pooling2d(x,ksize,strides):
    return tf.nn.max_pool(x,ksize,strides,padding="SAME")


x=tf.placeholder(shape=[None,784],dtype=tf.float32)
y=tf.placeholder(shape=[None,10],dtype=tf.float32)
x_image=tf.reshape(x,[-1,28,28,1])
keep=tf.placeholder(tf.float32)
#change 1:normalize input
mean,var=tf.nn.moments(x_image,[1,2],keep_dims=True)
x_image=tf.subtract(x_image,mean)
x_image=tf.divide(x_image,tf.sqrt(var))

#for change 2
def block1(x,block_name):

    mw1=get_weight_variable([3,3,16,32],block_name+"_mW1weights")

    mw21=get_weight_variable([1,5,16,32],block_name+"_mW2weights1")
    mw22=get_weight_variable([5,1,32,32],block_name+"_mW2weights2")
    mb2=get_bias_variable([32],block_name+"_mb2")
    mwn=get_weight_variable([3,3,48,32],block_name+"_mWnweights")

    module1_out=conv2d(x,mw1)
    module2_out=conv2d(x,mw21)
    module2_out=conv2d(module2_out,mw22)
    module2_out+=mb2
    module3_out=pooling2d(x,[1,5,5,1],[1,1,1,1])
    module_out=module1_out+module2_out
    module_out=tf.concat([module_out,module3_out],3)
    return conv2d(module_out,mwn)

# #init version
# def block1(x,block_name):
#
#     mw1=get_weight_variable([3,3,1,32],block_name+"_mW1weights")
#
#     mw21=get_weight_variable([1,5,1,32],block_name+"_mW2weights1")
#     mw22=get_weight_variable([5,1,32,32],block_name+"_mW2weights2")
#     mb2=get_bias_variable([32],block_name+"_mb2")
#     mwn=get_weight_variable([3,3,33,32],block_name+"_mWnweights")
#
#     module1_out=conv2d(x,mw1)
#     module2_out=conv2d(x,mw21)
#     module2_out=conv2d(module2_out,mw22)
#     module2_out+=mb2
#     module3_out=pooling2d(x,[1,5,5,1],[1,1,1,1])
#     module_out=module1_out+module2_out
#     module_out=tf.concat([module_out,module3_out],3)
#     return conv2d(module_out,mwn)
def block2(x,block_name):
    mw1=get_weight_variable([3,3,32,64],block_name+"_mW1weights")

    mw21=get_weight_variable([1,5,32,64],block_name+"_mW2weights1")
    mw22=get_weight_variable([5,1,64,64],block_name+"_mW2weights2")
    mb2=get_bias_variable([64],block_name+"_mb2")
    mwn=get_weight_variable([3,3,96,64],block_name+"_mWnweights")

    module1_out=conv2d(x,mw1,strides=[1,2,2,1])
    module2_out=conv2d(x,mw21)
    module2_out=conv2d(module2_out,mw22,strides=[1,2,2,1])
    module2_out+=mb2
    module3_out=pooling2d(x,[1,5,5,1],[1,2,2,1])
    module_out=module1_out+module2_out
    module_out=tf.concat([module_out,module3_out],3)
    return conv2d(module_out,mwn)




mnist=input_data.read_data_sets("C:\\Users\\rebel\\.keras\\datasets",one_hot=True)

sess=tf.InteractiveSession()
#change 2 add global activate conv
gpw=get_weight_variable([3,3,x_image.shape[3],16],"global_act_conv")
x_image=conv2d(x_image,gpw)
block1_out=block1(x_image,"block1")
block1_out=tf.nn.swish(block1_out,name="block_1_act")
block1_out=tf.nn.dropout(block1_out,keep)

block2_out=block2(block1_out,"block2")
block2_out=pooling2d(block2_out,[1,3,3,1],[1,2,2,1])
block2_out=tf.nn.swish(block2_out,name="block_2_act")
block2_out=tf.nn.dropout(block2_out,keep)

#change3 add glob avg pool  -- bad
# block_out=tf.nn.avg_pool(block2_out,[1,7,7,1],[1,1,1,1],padding="VALID")
# gcw=get_weight_variable([1,1,64,10],name="globalconv")
# block_out=tf.nn.conv2d(block_out,gcw,padding="VALID",strides=[1,1,1,1])

# change 4 use conv instead of avg pool
gcw1=get_weight_variable([7,7,64,1024],name="globalconv1")
block_out=tf.nn.conv2d(block2_out,gcw1,padding="VALID",strides=[1,1,1,1])
#activate function nuibi
block_out=tf.nn.swish(block_out)
gcw2=get_weight_variable([1,1,1024,10],name="globalconv2")
block_out=tf.nn.conv2d(block_out,gcw2,padding="VALID",strides=[1,1,1,1])
#don not add activation on thin fm
#block_out=tf.nn.swish(block_out)
block_out=tf.reshape(block_out,[-1,10])
y_out=tf.nn.softmax(block_out)

# # #final conv
# gcw=get_weight_variable([7,7,64,1024],name="globalconv")
# block_out=tf.nn.conv2d(block2_out,gcw,padding="VALID",strides=[1,1,1,1])
# block_out=tf.nn.swish(block_out)
# block_out=tf.reshape(block_out,[-1,1024])
#
# #softmax
# softmax_w=get_weight_variable([1024,10],"softmax_w")
# softmax_b=get_bias_variable([10],"softmax_b")
# y_out=tf.nn.softmax(tf.matmul(block_out,softmax_w)+softmax_b)


cross_en=tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_out),reduction_indices=[1]))
tran_step=tf.train.AdamOptimizer(1e-4).minimize(cross_en)
tran_step_sgd=tf.train.GradientDescentOptimizer(1e-14).minimize(cross_en)
corr=tf.equal(tf.argmax(y,1),tf.argmax(y_out,1))
accuracy=tf.reduce_mean(tf.cast(corr,tf.float32))

tf.global_variables_initializer().run()
for i in range(1000):
    #batch=mnist.train.next_batch(128)
    #multi batch size
    if i<900:
        batch=mnist.train.next_batch(128)
    else:
        batch=mnist.train.next_batch(512)
    if i%100==0:

        #print(y_out.eval(feed_dict={x:batch[0],y:batch[1],keep:1.0}))
        tran_acc=accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep:1.0})
        loss=cross_en.eval(feed_dict={x:batch[0],y:batch[1],keep:1.0})
        print("step: {} train_acc:{} loss:{}".format(i,tran_acc,loss))
    #keep_prob=0.7 acc 0.989
    #tran_step.run(feed_dict={x:batch[0],y:batch[1],keep:0.7})
    #multi optimizer
    if i<800:
        tran_step.run(feed_dict={x:batch[0],y:batch[1],keep:0.7})
    else:
        tran_step_sgd.run(feed_dict={x:batch[0],y:batch[1],keep:1.0})

acc_list=[]
for i in range(1,len(mnist.test.images)//1000):
    acc_list.append(accuracy.eval(feed_dict={x:mnist.test.images[(i-1)*1000:i*1000],y:mnist.test.labels[(i-1)*1000:i*1000],keep:1.0}))


print("test acc:{}".format(np.mean(acc_list)))






