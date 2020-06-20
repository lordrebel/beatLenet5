import tensorflow as tf
from cifar10_loader import Cifa10_data
import numpy as np
import time
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

cropSize=28
x=tf.placeholder(shape=[None,cropSize,cropSize,3],dtype=tf.float32)
y=tf.placeholder(shape=[None,10],dtype=tf.float32)
keep=tf.placeholder(tf.float32)
#change 1:normalize input
mean,var=tf.nn.moments(x,[1,2],keep_dims=True)
x_image__=tf.subtract(x,mean)
x_image1=tf.divide(x_image__,tf.sqrt(var))

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




cifar_10=Cifa10_data("C:\\Users\\rebel\\.keras\\datasets\\cifar-10-batches-py",256,0.05,0.1,cropSize)

sess=tf.InteractiveSession()
#change 2 add global activate conv
gpw=get_weight_variable([1,1,x_image1.shape[3],16],"global_act_conv")
x_image=conv2d(x_image1,gpw)
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
tran_step_warm=tf.train.AdamOptimizer(6e-4).minimize(cross_en)
tran_step=tf.train.AdamOptimizer(4e-5).minimize(cross_en)
tran_step_sgd=tf.train.GradientDescentOptimizer(4e-6).minimize(cross_en)
corr=tf.equal(tf.argmax(y,1),tf.argmax(y_out,1))
accuracy=tf.reduce_mean(tf.cast(corr,tf.float32))
steps=10000
tf.global_variables_initializer().run()
start =time.time()
for i in range(steps):
    batch=cifar_10.next_Batch_train()
    if i%500==0 and i!=0:
        validate_data=cifar_10.get_validate_datas()
        #print(y_out.eval(feed_dict={x:batch[0],y:batch[1],keep:1.0}))
        tran_acc=accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep:1.0})
        loss=cross_en.eval(feed_dict={x:batch[0],y:batch[1],keep:1.0})
        valid_acc=accuracy.eval(feed_dict={x:validate_data[0],y:validate_data[1],keep:1.0})
        print("step: {} train_acc:{} val_acc:{} loss:{}".format(i,tran_acc,valid_acc,loss))
    #keep_prob=0.7 acc 0.989
    #tran_step.run(feed_dict={x:batch[0],y:batch[1],keep:0.7})
    #multi optimizer

    if i<0.2*steps:
        tran_step_warm.run(feed_dict={x:batch[0],y:batch[1],keep:0.3}
                      )
    elif i<0.7*steps:
        tran_step.run(feed_dict={x:batch[0],y:batch[1],keep:0.5}
                                     )
    else:
        tran_step_sgd.run(feed_dict={x:batch[0],y:batch[1],keep:1.0}
                      )

end=time.time()
print("start to do testing....")
acc_list=[]
test_batch="dummy"
while test_batch is not None:
    test_batch=cifar_10.next_Batch_test()
    if test_batch is None:
        break
    acc=accuracy.eval(feed_dict={x:test_batch[0],y:test_batch[1],keep:1.0})
    acc_list.append(acc)


print("test acc:{}".format(np.mean(acc_list)))
print ("train cost time:{}".format((end-start)/60))






