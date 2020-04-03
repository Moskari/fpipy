import tensorflow as tf


@tf.function
def demosaic_bilin_float_tensorflow(cfa, masks):
    """Demosaics and computes (pseudo)radiance using given arrays.

    Parameters
    ----------
    cfa : tf.Tensor
        (y, x) array of CFA data.
    masks : tf.Tensor
        (3, y, x) boolean array of R, G, and B mask arrays.

    Returns
    -------
    rgb : tf.Tensor
        (3, y, x) demosaiced RGB image
    """
    g_krnl = tf.constant(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]], dtype=tf.float32) / 4.
    rb_krnl = tf.constant(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=tf.float32) / 4.

    rgbs = tf.cast(cfa, tf.float32) * tf.cast(masks, tf.float32)
    paddings = tf.constant([[0, 0], [1, 1], [1, 1]])
    rgbs = tf.pad(rgbs, paddings, mode='REFLECT')
    shape = rgbs.shape

    # Actually cross-correlation
    R = tf.nn.convolution(
        tf.reshape(rgbs[0, ...], [1, shape[1], shape[2], 1]),
        tf.reshape(rb_krnl, [3, 3, 1, 1]),
        strides=(1, 1, 1, 1),
        padding='VALID',
    )
    G = tf.nn.convolution(
        tf.reshape(rgbs[1, ...], [1, shape[1], shape[2], 1]),
        tf.reshape(g_krnl, [3, 3, 1, 1]),
        strides=(1, 1, 1, 1),
        padding='VALID',
    )
    B = tf.nn.convolution(
        tf.reshape(rgbs[2, ...], [1, shape[1], shape[2], 1]),
        tf.reshape(rb_krnl, [3, 3, 1, 1]),
        strides=(1, 1, 1, 1),
        padding='VALID',
    )

    return tf.stack([R[0, ..., 0], G[0, ..., 0], B[0, ..., 0]], axis=0)

if __name__ == '__main__':
    pass
