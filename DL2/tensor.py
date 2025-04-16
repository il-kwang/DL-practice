import tensorflow as tf

# 간단한 상수와 연산 예제
a = tf.constant(3)
b = tf.constant(4)
result = tf.add(a, b)
tf.print("3 + 4 =", result)


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
