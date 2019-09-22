import tensorflow as  tf

"""
remember to initialize your variables, create a session and run the operations inside the session. 
"""

y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

loss = tf.Variable((y - y_hat) ** 2, name='loss')

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

const_1 = tf.constant(10)
const_2 = tf.constant(20)
result = tf.multiply(const_1, const_2)
print(result)
session = tf.Session()
print(session.run(result))
