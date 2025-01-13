import tensorflow as tf

def focal_loss(gamma=2.0, alpha=5.75):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = - alpha_t * (1 - p_t) ** gamma * tf.math.log(p_t + 1e-5)
        return tf.reduce_mean(fl)
    return focal_loss_fixed

def tversky_loss(alpha=0.5):
    beta = 1 - alpha
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_pos = tf.reduce_sum(y_true * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)
        return 1 - (true_pos + 1) / (true_pos + alpha * false_neg + beta * false_pos + 1)
    return loss

def skeleton_loss(y_true, y_pred):
    inter = tf.reduce_sum(y_true * y_pred) / tf.reduce_sum(y_true)
    mse_score = tf.reduce_mean(tf.square(y_true - y_pred))
    mean_constraint = tf.abs(tf.reduce_mean(y_pred) - tf.reduce_mean(y_true))
    return 1.0 - inter + mse_score + 0.1 * mean_constraint


# - - - - - Loss depending on the objects skeleton - - - - - #

def skeleton_recall(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    recall = intersection / (tf.reduce_sum(y_true) + 1e-8)
    return 1 - recall

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)

def dual_dice_loss(y_true, y_pred):
    c1 = 0.3
    c2 = 1.0 - c1
    return c1 * dice_loss(y_true, y_pred) + c2 * dice_loss(1 - y_true, 1 - y_pred)

def bce_dice_loss(bce_coef=0.5):
    def bcl(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        return bce_coef * bce + (1.0 - bce_coef) * dice
    return bcl

def dice_skeleton_loss(skeleton_coef=0.5, bce_coef=0.5):
    bdl = bce_dice_loss(bce_coef)
    def _dice_skeleton_loss(y_true, y_pred):
        y_pred = tf.square(y_pred)
        return (1.0 - skeleton_coef) * bdl(y_true, y_pred) + skeleton_coef * skeleton_recall(y_true, y_pred)
    return _dice_skeleton_loss

def dsl(y_true, y_pred):
    return dice_skeleton_loss(0.5, 0.5)(y_true, y_pred)