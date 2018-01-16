import tensorflow as tf
from tensorflow.contrib import rnn

class model(object):

    def __init__(self,max_name_arr_len,max_cat_name_len,max_item_desc_len,vocab_size,emb_size):

        self.name_arr = tf.placeholder(dtype=tf.int32,shape=[None,max_name_arr_len],name='name_arr')
        self.name_arr_len = tf.placeholder(dtype=tf.int32,shape=[None],name='name_arr_len')
        self.item_cont_arr = tf.placeholder(dtype=tf.int32,shape=[None,1],name='item_cont_arr')
        self.cat_name_arr = tf.placeholder(dtype=tf.int32,shape=[None,max_cat_name_len],name='cat_name_arr')
        self.cat_name_arr_len = tf.placeholder(dtype=tf.int32,shape=[None],name='cat_name_arr_len')
        self.brand_name = tf.placeholder(dtype=tf.int32,shape=[None,1],name='brand_name')
        self.price = tf.placeholder(dtype=tf.float32,shape=[None,1],name='price')
        self.shipping_arr = tf.placeholder(dtype=tf.float32,shape=[None,1],name='shipping_arr')
        self.item_desc_arr = tf.placeholder(dtype=tf.int32,shape=[None,max_item_desc_len],name='item_desc_arr')
        self.item_desc_arr_len = tf.placeholder(dtype=tf.int32,shape=[None],name='item_desc_arr_len')
        self.hidden_Units = 50
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        with tf.name_scope('word-embedding-lookup'):
            self.embeddings = tf.get_variable(name='embeddings',
                                              initializer=tf.truncated_normal(
                                                  shape=[self.vocab_size,self.emb_size],mean=0.0,stddev=0.01,dtype=tf.float32))
            self.item_cond_embeddings = tf.get_variable(name='item_cond_embeddings',
                                              initializer=tf.truncated_normal(
                                                  shape=[5, 16], mean=0.0, stddev=0.01,
                                                  dtype=tf.float32))
            self.name_arr_emb = tf.nn.embedding_lookup(self.embeddings,self.name_arr)
            self.cat_name_arr_emb = tf.nn.embedding_lookup(self.embeddings , self.cat_name_arr)
            self.item_desc_arr_emb = tf.nn.embedding_lookup(self.embeddings , self.item_desc_arr)
            self.brand_name_emb = tf.nn.embedding_lookup(self.embeddings,self.brand_name)
            self.item_cont_arr_emb = tf.nn.embedding_lookup(self.item_cond_embeddings,self.item_cont_arr)
        ## name
        with tf.variable_scope('name-array'):
            lstm_cell_0 = rnn.BasicLSTMCell(num_units=self.hidden_Units)
            outputs_name, _ = tf.nn.dynamic_rnn(lstm_cell_0,inputs=self.name_arr_emb,sequence_length=self.name_arr_len,dtype=tf.float32)
            outputs_name = tf.reduce_max(outputs_name,axis=1)

        ##category name array
        with tf.variable_scope('category-name'):
            lstm_cell_1 = rnn.BasicLSTMCell(num_units=self.hidden_Units)
            outputs_cat_name , _ = tf.nn.dynamic_rnn(lstm_cell_1,inputs=self.cat_name_arr_emb,sequence_length=self.cat_name_arr_len, dtype=tf.float32)
            outputs_cat_name = tf.reduce_max(outputs_cat_name,axis=1)
        ## item description array
        with tf.variable_scope('item-description-array'):
            lstm_cell_2 = rnn.BasicLSTMCell(num_units=self.hidden_Units)
            outputs_item_desc_name , _ = tf.nn.dynamic_rnn(lstm_cell_2,inputs=self.item_desc_arr_emb,sequence_length=self.item_desc_arr_len,dtype=tf.float32)
            outputs_item_desc_name = tf.reduce_max(outputs_item_desc_name, axis=1)

        ## Stack all inputs
        # output = tf.concat([outputs_name[:,-1,:],outputs_cat_name[:,-1,:],
        #                     outputs_item_desc_name[:,-1,:],self.brand_name_emb[:,0,:],
        #                     self.item_cont_arr_emb,self.shipping_arr],
        #                    axis=1)
        output = tf.concat([outputs_name, outputs_cat_name,
                   outputs_item_desc_name, self.brand_name_emb[:, 0, :],
                   self.item_cont_arr_emb[:, 0, :], self.shipping_arr],
                  axis=1)
        output_dimension = 4*self.hidden_Units + 16 + 1
        hidden_dimension_1 = 50
        hidden_dimension_2 = 20
        hidden_dimension_3 = 1
        ## 2 hidden  layers
        W_1 = tf.get_variable("W_1",shape=[output_dimension,hidden_dimension_1],initializer=tf.contrib.layers.xavier_initializer())
        b_1 = tf.Variable(initial_value=tf.constant(value=0.01,shape=[hidden_dimension_1]))
        layer_1_out = tf.nn.relu(tf.matmul(output,W_1) + b_1)

        W_2 = tf.get_variable("W_2",shape=[hidden_dimension_1,hidden_dimension_2],initializer=tf.contrib.layers.xavier_initializer())
        b_2 = tf.Variable(initial_value=tf.constant(value=0.01,shape=[hidden_dimension_2]))
        layer_2_out = tf.nn.relu(tf.matmul(layer_1_out,W_2) + b_2)

        W_3 = tf.get_variable("W_3",shape=[hidden_dimension_2,hidden_dimension_3],initializer=tf.contrib.layers.xavier_initializer())
        b_3 = tf.Variable(initial_value=tf.constant(value=0.01,shape=[hidden_dimension_3]))
        self.final_score = tf.matmul(layer_2_out,W_3) + b_3

        ## loss
        self.loss = tf.losses.mean_squared_error(labels=self.price,predictions = self.final_score)

# m = model(max_name_arr_len=5,max_cat_name_len=3,max_item_desc_len=15,vocab_size=123,emb_size=50)