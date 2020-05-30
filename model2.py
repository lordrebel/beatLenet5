import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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

#test depth-seperatable
def depth_seper_conv3x3(x,layer_name):
    x_s=tf.shape(x)
    dw=get_weight_variable([3,3,x_s[3],1],layer_name+"3*3_dw")
    pw=get_weight_variable([1,1,x_s[3],x_s*2],layer_name+"3*3_pw")

    res=tf.nn.depthwise_conv2d(x,dw,[1,1,1,1],padding="SAME")
    res=tf.nn.conv2d(res,pw,[1,1,1,1],padding="SAME")
    return res

def depth_seper_conv5x5(x,layer_name):
    x_s=tf.shape(x)
    dw1=get_weight_variable([5,1,x_s[3],1],layer_name+"5*5_dw_1")
    dw2=get_weight_variable([1,5,x_s[3],1],layer_name+"5*5_dw_1")
    pw=get_weight_variable([1,1,x_s[3],x_s[3]*2],layer_name+"5*5_pw")

    res=tf.nn.depthwise_conv2d(x,dw1,[1,1,1,1],padding="SAME")
    res=tf.nn.depthwise_conv2d(res,dw2,[1,1,1,1],padding="SAME")
    res=tf.nn.conv2d(res,pw,[1,1,1,1],padding="SAME")
    return res


def depth_seper_conv7x7(x,layer_name):
    x_s=tf.shape(x)
    dw1=get_weight_variable([7,1,x_s[3],1],layer_name+"7*7_dw_1")
    dw2=get_weight_variable([1,7,x_s[3],1],layer_name+"7*7_dw_1")
    pw=get_weight_variable([1,1,x_s[3],x_s*2],layer_name+"7*7_pw")

    res=tf.nn.depthwise_conv2d(x,dw1,[1,1,1,1],padding="SAME")
    res=tf.nn.depthwise_conv2d(res,dw2,[1,1,1,1],padding="SAME")
    res=tf.nn.conv2d(res,pw,[1,1,1,1],padding="SAME")
    return res

def depth_block(x,block_name):
    m3_3=depth_seper_conv3x3(x,block_name)
    m5_5=depth_seper_conv3x3(x,block_name)
    m7_7=depth_seper_conv7x7(x,block_name)
    x_s=tf.shape(x)
    mwn=get_weight_variable([3,3,4*x_s[3],2*x_s[3]],block_name+"_mWnweights")
    mm_3_3=pooling2d(x,[1,3,3,1],[1,1,1,1])
    mm_5_5=pooling2d(x,[1,5,5,1],[1,1,1,1])
    mm_7_7=pooling2d(x,[1,7,7,1],[1,1,1,1])
    m_out=m3_3+m5_5+m7_7
    mm_out=mm_3_3+mm_5_5+mm_7_7
    module_out=tf.concat([m_out,mm_out],3)
    return conv2d(module_out,mwn,[1,1,1,1])

x=tf.placeholder(shape=[None,784],dtype=tf.float32)
y=tf.placeholder(shape=[None,10],dtype=tf.float32)
x_image=tf.reshape(x,[-1,28,28,1])
keep=tf.placeholder(tf.float32)

mean,var=tf.nn.moments(x_image,[1,2],keep_dims=True)
x_image=tf.subtract(x_image,mean)
x_image=tf.divide(x_image,var)