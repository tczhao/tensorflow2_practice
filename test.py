import tensorflow as tf

print("--Basic arithmatic--------------------")

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
print(C)

random_float = tf.random.uniform(shape=())
print(random_float)

zero_vector = tf.zeros(shape=(2))
print(zero_vector)

print(zero_vector.shape)
print(zero_vector.dtype)
print(zero_vector.numpy())

print("--Gradient calculation---------------------")
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y, x)
print(y, y_grad)

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])
print(L, w_grad, b_grad)

print("--Linear regression---------------")

print("numpy")
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

a, b = 0, 0

num_epoch = 10000
learning_rate = 5e-4
for e in range(num_epoch):
    y_pred = a * X + b
    grad_a, grad_b = 2 * (y_pred - y).dot(X), 2 * (y_pred - y).sum()
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)

print("tf")
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print("--Model and training")
# model structure: tf.keras.Model 和 tf.keras.layers
# model loss: tf.keras.losses
# model optimizer: tf.keras.optimizer
# model evaluation: tf.keras.metrics
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output

model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)

print("--MLP keras example")
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

num_epochs = 0.1
batch_size = 50
learning_rate = 0.001

model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())

print("--CNN keras example")
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

model = CNN()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())

print("--MobileNet")
import tensorflow_datasets as tfds

num_epoch = 5
batch_size = 50
learning_rate = 0.001

dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)
model = tf.keras.applications.MobileNetV2(weights=None, classes=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for e in range(num_epoch):
    for images, labels in dataset:
        with tf.GradientTape() as tape:
            labels_pred = model(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
            loss = tf.reduce_mean(loss)
            print("loss %f" % loss.numpy())
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    print(labels_pred)

print("--RNN-----------------")
class DataLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt',
            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index+seq_length])
            next_char.append(self.text[index+seq_length])
        return np.array(seq), np.array(next_char)    # [batch_size, seq_length], [num_batch]

class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        inputs = tf.one_hot(inputs, depth=self.num_chars)       # [batch_size, seq_length, num_chars]
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)
        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)

num_batches = 1000
seq_length = 40
batch_size = 50
learning_rate = 1e-3

data_loader = DataLoader()
model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(seq_length, batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

def predict(self, inputs, temperature=1.):
    batch_size, _ = tf.shape(inputs)
    logits = self(inputs, from_logits=True)
    prob = tf.nn.softmax(logits / temperature).numpy()
    return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                        for i in range(batch_size.numpy())])

X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    X = X_
    print("diversity %f:" % diversity)
    for t in range(400):
        y_pred = model.predict(X, diversity)
        print(data_loader.indices_char[y_pred[0]], end='', flush=True)
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
    print("\n")

print("--DRL-------------------")
# STEPS
# import gym

# env = gym.make('CartPole-v1')       # make game environment with name CartPole-v1
# state = env.reset()                 # init environment and its state
# while True:
#     env.render()                    # render to current environment
#     action = model.predict(state)   # predict the next actino giving current state
#     next_state, reward, done, info = env.step(action)   # let environment action and obtain the next state, reward and ending status
#     if done:                        # break if game end
#         break

import numpy as np
import gym
import random
from collections import deque

num_episodes = 500              # 游戏训练的总episode数量
num_exploration_episodes = 100  # 探索过程所占的episode数量
max_len_episode = 1000          # 每个episode的最大回合数
batch_size = 32                 # 批次大小
learning_rate = 1e-3            # 学习率
gamma = 1.                      # 折扣因子
initial_epsilon = 1.            # 探索起始时的探索率
final_epsilon = 0.01            # 探索终止时的探索率

class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)

env = gym.make('CartPole-v1')       # 实例化一个游戏环境，参数为游戏名称
model = QNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
replay_buffer = deque(maxlen=10000) # 使用一个 deque 作为 Q Learning 的经验回放池
epsilon = initial_epsilon
for episode_id in range(num_episodes):
    state = env.reset()             # 初始化环境，获得初始状态
    epsilon = max(                  # 计算当前探索率
        initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
        final_epsilon)
    for t in range(max_len_episode):
        env.render()                                # 对当前帧进行渲染，绘图到屏幕
        if random.random() < epsilon:               # epsilon-greedy 探索策略，以 epsilon 的概率选择随机动作
            action = env.action_space.sample()      # 选择随机动作（探索）
        else:
            action = model.predict(np.expand_dims(state, axis=0)).numpy()   # 选择模型计算出的 Q Value 最大的动作
            action = action[0]

        # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
        next_state, reward, done, info = env.step(action)
        # 如果游戏Game Over，给予大的负奖励
        reward = -10. if done else reward
        # 将(state, action, reward, next_state)的四元组（外加 done 标签表示是否结束）放入经验回放池
        replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
        # 更新当前 state
        state = next_state

        if done:                                    # 游戏结束则退出本轮循环，进行下一个 episode
            print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
            break

        if len(replay_buffer) >= batch_size:
            # 从经验回放池中随机取一个批次的四元组，并分别转换为 NumPy 数组
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                *random.sample(replay_buffer, batch_size))
            batch_state, batch_reward, batch_next_state, batch_done = \
                [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
            batch_action = np.array(batch_action, dtype=np.int32)

            q_value = model(batch_next_state)
            y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)  # 计算 y 值
            with tf.GradientTape() as tape:
                loss = tf.keras.losses.mean_squared_error(  # 最小化 y 和 Q-value 的距离
                    y_true=y,
                    y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=2), axis=1)
                )
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))       # 计算梯度并更新参数

#################################################################################################
# Modules
# checkpoints saves the paramters but requires original code to restore
# checkpoint manager to keep only the latest checkpoint
# savemodel saves both the model structure but not param 

# example
# # train.py 模型训练阶段
# model = MyModel()
# # 实例化Checkpoint，指定保存对象为model（如果需要保存Optimizer的参数也可加入）
# checkpoint = tf.train.Checkpoint(myModel=model)
# # ...（模型训练代码）
# # 模型训练完毕后将参数保存到文件（也可以在模型训练过程中每隔一段时间就保存一次）
# checkpoint.save('./save/model.ckpt')

# # test.py 模型使用阶段
# model = MyModel()
# checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
# checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数
# # 模型使用代码


#################################################################################
# Distributed training

# Single Machine, multi card
# MirroredStrategy 的步骤如下：

# 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型；

# 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）；

# N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度；

# 使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和；

# 使用梯度求和的结果更新本地变量（镜像变量）；

# 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）。

# 默认情况下，TensorFlow 中的 MirroredStrategy 策略使用 NVIDIA NCCL 进行 All-reduce 操作。


# Multimachine

# MultiWorkerMirroredStrategy

# cluster 说明了整个多机集群的结构和每台机器的网络地址（IP + 端口号）。对于每一台机器，cluster 的值都是相同的；

# task 说明了当前机器的角色。例如， {'type': 'worker', 'index': 0} 说明当前机器是 cluster 中的第 0 个 worker（即 localhost:20000 ）。每一台机器的 task 值都需要针对当前主机进行分别的设置。

# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {
#         'worker': ["localhost:20000", "localhost:20001"]
#     },
#     'task': {'type': 'worker', 'index': 0}
# })

# parallel map
# dataset prefetch
# allows maximise cpu and gpu utilisation for data preprocessing