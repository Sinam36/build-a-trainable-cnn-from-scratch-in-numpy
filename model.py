"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix,axis = 1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix,axis = 1,keepdims = True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix,axis = 1,keepdims = True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    a = row_max(logits)
    b = logits - a
    return np.exp(b)

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    a = exp_shifted(logits)
    b = row_sum(a)
    return a/b

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    a = np.zeros((labels.size,num_classes))
    b = np.arange(0,labels.size)
    a[b,labels] = 1.0
    return a

# Step 7 - gather_true_class_probs
import numpy as np
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    a = np.arange(0,labels.size)
    return probs[a,labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    p = np.clip(gather_true_class_probs(probs,labels),eps,None)
    return (np.sum(np.log(p))/labels.size)*(-1)

# Step 9 - accuracy
import numpy as np
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    a = argmax_rows(logits_or_probs)
    return np.mean(a == labels)

# Step 10 - he_std
import numpy as np
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
import numpy as np
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    he_std_dev = he_std(fan_in)
    return np.random.normal(loc = 0.0,scale = he_std_dev,size = shape)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length)

# Step 13 - pad_2d
import numpy as np
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    return np.pad(images,pad_width = ((0,0),(0,0),(pad,pad),(pad,pad)),mode = 'constant')

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return (input_size + 2*padding - kernel)//stride + 1

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.
    a = pad_2d(images,padding)
    out_h = output_spatial_size(images.shape[2],kernel_h,stride,padding)
    out_w = output_spatial_size(images.shape[3],kernel_w,stride,padding)
    view = np.lib.stride_tricks.sliding_window_view(a,window_shape = (kernel_h,kernel_w),axis = (2,3))
    view = view[:,:,::stride,::stride,:,:]
    view = view.transpose(0,2,3,1,4,5)
    return view.reshape(-1,images.shape[1]*kernel_h*kernel_w)

# Step 16 - col2im
import numpy as np
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    N, C, H, W = input_shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    img_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
    cols_reshaped = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    for h in range(out_h):
        for w in range(out_w):
            row_start = h * stride
            row_end = row_start + kernel_h
            col_start = w * stride
            col_end = col_start + kernel_w
            img_padded[:, :, row_start:row_end, col_start:col_end] += cols_reshaped[:, h, w, :, :, :]
    if padding > 0:
        return img_padded[:, :, padding:-padding, padding:-padding]
    
    return img_padded

# Step 17 - conv2d_forward
import numpy as np
def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.
    N, C_in, H, W = x.shape
    C_out, _, kernel_h, kernel_w = weights.shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1
    cols = im2col(x, kernel_h, kernel_w, stride, padding)
    reshaped_weights = weights.reshape(C_out, -1)
    out_col = cols @ reshaped_weights.T + bias
    out_reshaped = out_col.reshape(N, out_h, out_w, C_out)
    out = out_reshaped.transpose(0, 3, 1, 2)
    cache = {
        'x_shape': x.shape,
        'weights': weights,
        'cols': cols,
        'stride': stride,
        'padding': padding,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w
    }
    
    return out, cache

# Step 18 - conv2d_grad_input
import numpy as np
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im
    x_shape = cache['x_shape']
    weights = cache['weights']
    stride = cache['stride']
    padding = cache['padding']
    kernel_h = cache['kernel_h']
    kernel_w = cache['kernel_w']
    
    N, C_out, out_h, out_w = d_out.shape
    C_out, C_in, _, _ = weights.shape
    dout_reshaped = d_out.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, C_out)
    weights_reshaped = weights.reshape(C_out, -1)
    d_cols = dout_reshaped @ weights_reshaped
    d_x = col2im(d_cols, x_shape, kernel_h, kernel_w, stride, padding)
    
    return d_x

# Step 19 - conv2d_grad_weights
import numpy as np
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    cols = cache['cols']
    weights_shape = cache['weights'].shape
    N, C_out, out_h, out_w = d_out.shape
    dout_reshaped = d_out.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, C_out)
    dW_flat = dout_reshaped.T @ cols
    dW = dW_flat.reshape(weights_shape)
    return dW

# Step 20 - conv2d_grad_bias
import numpy as np
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes.
    db = np.sum(d_out, axis=(0, 2, 3))
    return db

# Step 21 - conv2d_backward
import numpy as np
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)
    return (dx, dW, db)

# Step 22 - maxpool2d_forward
import numpy as np
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    N, C, H, W = x.shape
    out_h = output_spatial_size(H, kernel, stride, padding=0)
    out_w = output_spatial_size(W, kernel, stride, padding=0)
    windows = np.lib.stride_tricks.sliding_window_view(x, (kernel, kernel), axis=(2, 3))
    strided_windows = windows[:, :, ::stride, ::stride, :, :]
    flat_windows = strided_windows.reshape(N, C, out_h, out_w, kernel * kernel)
    out = np.max(flat_windows, axis=-1)
    argmax = np.argmax(flat_windows, axis=-1)
    cache = {
        'x_shape': x.shape,
        'argmax': argmax,
        'kernel': kernel,
        'stride': stride
    }
    
    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    window_flat = np.zeros(kernel * kernel)
    window_flat[argmax_index] = grad_value
    window_2d = window_flat.reshape(kernel, kernel)
    
    return window_2d

# Step 24 - maxpool2d_backward
import numpy as np
def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window
    x_shape = cache['x_shape']
    argmax = cache['argmax']
    kernel = cache['kernel']
    stride = cache['stride']
    dx = np.zeros(x_shape)
    N, C, out_h, out_w = d_out.shape
    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    grad_value = d_out[n, c, h, w]
                    idx = argmax[n, c, h, w]
                    window_grad = scatter_grad_window(grad_value, idx, kernel)
                    h_start = h * stride
                    h_end = h_start + kernel
                    w_start = w * stride
                    w_end = w_start + kernel
                    dx[n, c, h_start:h_end, w_start:w_end] += window_grad
                    
    return dx

# Step 25 - relu_forward
import numpy as np
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    out = np.maximum(0, x)
    cache = {'x': x}
    return out, cache

# Step 26 - relu_backward
import numpy as np
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x = cache['x']
    dx = d_out * (x > 0)
    return dx

# Step 27 - flatten_forward
import numpy as np
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    cache = {'x_shape': x.shape}
    N = x.shape[0]
    out = x.reshape(N, -1)
    
    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    x_shape = cache['x_shape']
    dx = d_out.reshape(x_shape)
    
    return dx

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.
    out = x @ weights + bias
    cache = {
        'x': x,
        'weights': weights
    }
    return out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    weights = cache['weights']
    dx = d_out @ weights.T
    
    return dx

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    dW = x.T @ dout
    
    return dW

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    db = np.sum(dout, axis=0)
    
    return db

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
  
    x = cache['x']
    
    dx = linear_grad_input(dout, cache)
    dW = linear_grad_weights(x, dout)
    db = linear_grad_bias(dout)
    
    return (dx, dW, db)

# Step 34 - softmax_cross_entropy_forward
import numpy as np
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    probs = stable_softmax(logits)
    
    losses = cross_entropy_loss(probs, y)
    
    mean_loss = float(np.mean(losses))
    
    return mean_loss

# Step 35 - softmax_cross_entropy_backward
import numpy as np
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    N, C = logits.shape
    
    probs = stable_softmax(logits)
    
    Y_true = one_hot(y, C) 
    
    dlogits = probs - Y_true
    
    dlogits = dlogits / N
    
    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    updated_param = param - (lr * grad)
    
    return updated_param

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    new_m = (beta_one * m) + ((1.0 - beta_one) * grad)
    
    return new_m

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    new_v = (beta_two * v) + ((1.0 - beta_two) * (grad ** 2))
    
    return new_v

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    corrected_moment = moment / (1.0 - (beta ** t))
    
    return corrected_moment

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    denom = np.sqrt(v_hat) + eps
    
    update_step = (lr * m_hat) / denom

    new_param = param - update_step
    
    return new_param

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)

    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)
    
    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)
    
    return new_param, new_m, new_v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    
    fan_in = in_channels * kernel_size * kernel_size
    
    W = he_init(shape, fan_in, seed=seed)
    
    b = init_zero_bias(out_channels)

    return {'W': W, 'b': b}

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    shape = (in_features, out_features)

    fan_in = in_features
    
    W = he_init(shape, fan_in, seed=seed)
    
    b = init_zero_bias(out_features)
    
    return {'W': W, 'b': b}

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    conv1 = init_conv_layer(out_channels=6, in_channels=in_channels, kernel_size=5, seed=seed)
    
    conv2 = init_conv_layer(out_channels=16, in_channels=6, kernel_size=5, seed=seed + 1)

    flattened_dim = 16 * 4 * 4

    fc1 = init_linear_layer(in_features=flattened_dim, out_features=120, seed=seed + 2)
    
    fc2 = init_linear_layer(in_features=120, out_features=num_classes, seed=seed + 3)
    
    return {
        'conv1': conv1,
        'conv2': conv2,
        'fc1': fc1,
        'fc2': fc2
    }

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
    out1, conv_cache = conv2d_forward(x, W, b, stride, pad)

    out2, relu_cache = relu_forward(out1)
    
    out3, pool_cache = maxpool2d_forward(out2, pool_size, stride=pool_size)

    cache_dict = {
        'conv_cache': conv_cache,
        'relu_cache': relu_cache,
        'pool_cache': pool_cache
    }
    
    return out3, cache_dict

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
    out_flat, flatten_cache = flatten_forward(x)
    
    out_fc1, fc1_cache = linear_forward(out_flat, fc1['W'], fc1['b'])
    
    out_relu, relu_cache = relu_forward(out_fc1)
    
    logits, fc2_cache = linear_forward(out_relu, fc2['W'], fc2['b'])
    
    cache_dict = {
        'flatten_cache': flatten_cache,
        'fc1_cache': fc1_cache,
        'relu_cache': relu_cache,
        'fc2_cache': fc2_cache
    }
    
    return logits, cache_dict

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    out1, cache1 = forward_conv_block(
        x, params['conv1']['W'], params['conv1']['b'], 
        pool_size=2, stride=1, pad=0
    )
    

    out2, cache2 = forward_conv_block(
        out1, params['conv2']['W'], params['conv2']['b'], 
        pool_size=2, stride=1, pad=0
    )
  
    logits, cache3 = forward_classifier_block(out2, params['fc1'], params['fc2'])

    cache_dict = {
        'block1': cache1,
        'block2': cache2,
        'classifier': cache3
    }
    
    return logits, cache_dict

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    conv_cache = cache['conv_cache']
    relu_cache = cache['relu_cache']
    pool_cache = cache['pool_cache']
    
    dx_pool = maxpool2d_backward(dout, pool_cache)
 
    dx_relu = relu_backward(dx_pool, relu_cache)
    
    dx, dw, db = conv2d_backward(dx_relu, conv_cache)
    
    return dx, dw, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # TODO: backprop through fc2 -> relu -> fc1 -> flatten using the cached values
    dx_fc2, dw_fc2, db_fc2 = linear_backward(dlogits, cache['fc2_cache'])
    
    dx_relu = relu_backward(dx_fc2, cache['relu_cache'])
  
    dx_fc1, dw_fc1, db_fc1 = linear_backward(dx_relu, cache['fc1_cache'])

    dx = flatten_backward(dx_fc1, cache['flatten_cache'])
    
    return {
        'dx': dx,
        'fc1': {'dW': dw_fc1, 'db': db_fc1},
        'fc2': {'dW': dw_fc2, 'db': db_fc2}
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients
    classifier_grads = backward_classifier_block(dlogits, caches['classifier'])
    
    dx2, dw2, db2 = backward_conv_block(classifier_grads['dx'], caches['block2'])
    
    dx1, dw1, db1 = backward_conv_block(dx2, caches['block1'])
  
    grads = {
        'conv1': {'dW': dw1, 'db': db1},
        'conv2': {'dW': dw2, 'db': db2},
        'fc1': classifier_grads['fc1'],
        'fc2': classifier_grads['fc2']
    }
    
    return grads

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    logits, _ = lenet_forward(x, params)
    
    predictions = np.argmax(logits, axis=1)
    
    return predictions

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    # TODO: Return (x, y) for a reproducible synthetic NCHW image dataset.
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=num_samples)
    x = rng.standard_normal(size=(num_samples, in_channels, image_size, image_size))
    shifts = y - (num_classes - 1) / 2
    x += shifts[:, np.newaxis, np.newaxis, np.newaxis]
    return x, y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    # TODO: return a reproducible permutation of [0, n) as an int ndarray of shape (n,).
    np.random.seed(seed)
    indices = np.random.permutation(n)
    return indices

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    # TODO: partition x and y into train and test halves using a shared shuffled order.
    n = x.shape[0]
    indices = shuffle_indices(n, seed=seed)
    split = int(n * test_fraction)
    test_idx = indices[:split]
    train_idx = indices[split:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    return x_train, y_train, x_test, y_test

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    # TODO: yield shuffled mini-batches of features and labels for one epoch of training.
    n = x.shape[0]
    indices = shuffle_indices(n, seed=seed)
    for i in range(0, n - batch_size + 1, batch_size):
        batch_idx = indices[i : i + batch_size]
        xb = x[batch_idx]
        yb = y[batch_idx]
        yield xb, yb

# Step 56 - train_step
def train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step):
    # TODO: Run forward + loss + backward and apply one Adam update to every parameter.
    logits, caches = lenet_forward(xb, params)
    loss = softmax_cross_entropy_forward(logits, yb)
    dlogits = softmax_cross_entropy_backward(logits, yb)
    grads = lenet_backward(dlogits, caches)
    
    new_params = {}
    new_opt_state = {}

    for layer in params:
        new_params[layer] = {}
        new_opt_state[layer] = {}
        
        for p_name in params[layer]:
            grad_key = 'd' + p_name
            param = params[layer][p_name]
            grad = grads[layer][grad_key]
            m = opt_state[layer][p_name]['m']
            v = opt_state[layer][p_name]['v']
            new_param, new_m, new_v = adam_step(
                param, grad, m, v, step, lr, beta_one, beta_two, eps
            )
            new_params[layer][p_name] = new_param
            new_opt_state[layer][p_name] = {'m': new_m, 'v': new_v}
            
    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    # TODO: iterate minibatches and apply one train_step per batch, tracking losses and step_counter.
    losses = []
    for xb, yb in iterate_minibatches(x, y, batch_size, seed=seed):
        step_counter += 1
        new_params, new_opt_state, batch_loss = train_step(
            params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step_counter
        )
        params = new_params
        opt_state = new_opt_state
        losses.append(batch_loss)
    return params, opt_state, step_counter, losses

# Step 58 - train_loop
def train_loop(params, x_train, y_train, num_epochs, batch_size, lr=1e-3, beta_one=0.9, beta_two=0.999, eps=1e-8, seed=0):
    # TODO: initialize Adam state, loop epochs calling train_one_epoch, return (params, loss_history).
    opt_state = {}
    for layer_name, layer_params in params.items():
        opt_state[layer_name] = {}
        for p_name, p_array in layer_params.items():
            opt_state[layer_name][p_name] = {
                'm': np.zeros_like(p_array),
                'v': np.zeros_like(p_array)
            }
    global_step_counter = 0
    full_loss_history = []
    for epoch in range(num_epochs):
        params, opt_state, global_step_counter, epoch_losses = train_one_epoch(
            params, opt_state, x_train, y_train, batch_size, 
            lr, beta_one, beta_two, eps, global_step_counter, seed=epoch
        )
        full_loss_history.extend(epoch_losses)
    return params, full_loss_history

# Step 59 - evaluate
def evaluate(params, x, y):
    # TODO: return the fraction of samples whose predicted class equals the label.
    predictions = lenet_predict(x, params)
    correct_matches = (predictions == y)
    accuracy = np.mean(correct_matches)
    return float(accuracy)

