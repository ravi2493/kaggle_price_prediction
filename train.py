from preprocessor import batch_iter,load_train_data
import tensorflow as tf
from model import model
import pickle
f = open(r'C:\Users\pravi\PycharmProjects\price_prediction\data_pickle','rb')
batch_size = 128
learning_rate = 0.001
# data,mp = load_train_data(path=r'C:\Users\pravi\Downloads\train.tsv')
data ,mp = pickle.load(f)
f.close()
epochs = 100
batches = batch_iter(data,batch_size=batch_size,epochs=epochs,Isshuffle=False)

# Define Training procedure
m = model(max_name_arr_len=mp['max_name_len'],max_cat_name_len=mp['max_cat_name_len'],
          max_item_desc_len=mp['max_item_desc_len'],vocab_size=mp['vocab_size'],emb_size=50)
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(m.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)

sess = tf.Session(config=session_conf)
saver = tf.train.Saver()
## intialize


sess.run(tf.global_variables_initializer())


print("done")
i = 0
for batch in batches:
    name_arr, name_arr_len, item_cont_arr, \
    cat_name_arr,cat_name_arr_len, brand_name_arr,\
    price_arr, shipping_arr, item_desc_arr, \
    item_desc_arr_len   = zip(*batch)

    feed_dict = {

    m.name_arr             : name_arr,
    m.name_arr_len         : name_arr_len,
    m.item_cont_arr        : item_cont_arr,
    m.cat_name_arr         : cat_name_arr,
    m.cat_name_arr_len     : cat_name_arr_len,
    m.brand_name           : brand_name_arr,
    m.shipping_arr         : shipping_arr,
    m.item_desc_arr        : item_desc_arr,
    m.item_desc_arr_len    : item_desc_arr_len,
    m.price                : price_arr
    }

    _,loss = sess.run([train_op,m.loss],feed_dict=feed_dict)
    i += 1
    print("step - "+ str(i) +" loss is " + str(loss))