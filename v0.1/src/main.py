import os
import numpy as np
import copy as cp
import model
from utils import *
import time
import cv2
import tensorflow as tf
import tensorlayer as tl


#PARAMETERS
N_EPOCH = 100
BATCH_SIZE = 16
BLOCK_SIZE = BATCH_SIZE * 20
INPUT_SIZE = 256
LEARNING_RATE = 2e-4
ADAM_BETA1 = 0.5
MODEL_SAVING_PER_EPOCH = 2
TEST_SAVING_PER_BLOCK = 10
TEST_SAVING_PER_EPOCH = 1
MEAN_ERROR_SCALING = 100

RGB_DIR = 'imgs/'
SKCH_DIR = 'sketches/'

TEST_SKCH_DIR = 'test/sketches/'
TEST_HINT_DIR = 'test/hints/'
TEST_RESULT_SAVING = 'test_results/'

MODEL_SAVING = 'models/'

def get_file_list(dir):
    ret_list = []
    for cur, dirs, files in os.walk(dir):
        ret_list = [os.path.join(dir, f) for f in files]
    return ret_list

def load_images(img_paths, mode = 1):
    ret = None
    if mode == 1:
        ret = np.array([cv2.imread(f) for f in img_paths]) / 127.5 - 1
    elif mode == 0:
        row, col = cv2.imread(img_paths[0], 0).shape
        ret = np.array([cv2.imread(f, 0).reshape(row, col, 1) for f in img_paths]) / 127.5 - 1

    return ret

def load_clipped_images(rgb_paths, skch_paths, modes):
    rgbs, skches = [], []
    for i in range(len(rgb_paths)):
        rgb = load_images([rgb_paths[i]], modes[0])[0]
        skch = load_images([skch_paths[i]], modes[1])[0]
        r, c = rgb.shape[:2]
        x = np.random.choice(r - INPUT_SIZE)
        y = np.random.choice(c - INPUT_SIZE)
        rgb = clip_img(rgb, (x, y))
        skch = clip_img(skch, (x, y))
        rgbs.append(cp.deepcopy(rgb))
        skches.append(cp.deepcopy(skch))
    return np.array(rgbs), np.array(skches)

def recover_images(imgs, cvt_color = True):
    if cvt_color:
        retimgs = np.array([cv2.cvtColor(i.astype(np.float32), cv2.COLOR_BGR2RGB) for i in imgs])
    else:
        retimgs = cp.deepcopy(imgs)
    return (retimgs + 1) * 127.5

def get_hints(skches, rgbs, n = 30, size = 10):
    hints = np.array([cv2.cvtColor(i.reshape(INPUT_SIZE, INPUT_SIZE).astype(np.float32), cv2.COLOR_GRAY2BGR) for i in skches])
    for i in range(len(hints)):
        for j in range(n):
            x = np.random.choice(INPUT_SIZE - size)
            y = np.random.choice(INPUT_SIZE - size)
            hints[i, x:x + size, y:y + size, :] = rgbs[i, x:x+size, y:y+size, :]
    return hints

def clip_img(img, pos = None):
    if pos == None:
        r, c, ch = img.shape
        x = np.random.choice(r - INPUT_SIZE)
        y = np.random.choice(c - INPUT_SIZE)
    else:
        x, y = pos
    return img[x:x+INPUT_SIZE, y:y+INPUT_SIZE, :]

def input_makeup(imgs):
    shape = imgs.shape
    n = 0
    if shape[0] % BATCH_SIZE != 0:
        n = BATCH_SIZE - shape[0] % BATCH_SIZE
        makeup = np.zeros([n] + list(shape[1:]))
        imgs = np.concatenate((imgs, makeup), axis=0)
    return imgs, n

def blur(imgs):
    return np.array([cv2.GaussianBlur(i, (21, 21), 10) for i in imgs])

def save_tests(gan, sess, imgs, dir, name_prefix, makeup = -1, mode = 'multiple', blurred = True):
    cnt = 0
    tot = len(imgs[0])
    for skch_test, hint_test in minibatches(imgs, BATCH_SIZE, False):
        cnt += 1
        tot -= BATCH_SIZE
        hints_input = cp.deepcopy(hint_test)
        if blurred:
            hints_input = blur(hints_input)
        feed_dict = {gan.hint_imgs: hints_input, gan.skch_imgs: skch_test}
        gens = sess.run(gan.g_net_eval.outputs, feed_dict = feed_dict)
        if mode == 'multiple':
            gens = recover_images(gens)
            hints = recover_images(hint_test)
            shape = list(hints.shape)
            shape[0] *= 2
            png = np.ones(shape)
            for i in range(0, len(png), 2):
                png[i] = hints[i // 2]
                png[i + 1] = gens[i // 2]
            r = len(png) // 4
            if tot == 0 and makeup != -1:
                r = (BATCH_SIZE - makeup) * 2 // 4 + (1 if ((BATCH_SIZE - makeup) * 2) % 4 != 0 else 0)
            tl.visualize.save_images(png[:r*4], (r, 4), os.path.join(dir, name_prefix + '_' + str(cnt) + '.png'))
        elif mode == 'single':
            gens = recover_images(gens, cvt_color = False)
            n = (BATCH_SIZE - makeup) if tot == 0 else BATCH_SIZE
            for i in range(n):
                path = os.path.join(dir, name_prefix + '_' + str(cnt) + '_' + str(i) + '.png')
                cv2.imwrite(path, gens[i])

def train(loadModel = False, model_paths = [], init_epoch = 0):
    gan = model.Model(INPUT_SIZE, BATCH_SIZE, MEAN_ERROR_SCALING, LEARNING_RATE, ADAM_BETA1)
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    if loadModel:
        gan.load_model(model_paths, sess)

    rgb_list = get_file_list(RGB_DIR)
    skch_list = get_file_list(SKCH_DIR)

    test_hint_imgs = load_images(get_file_list(TEST_HINT_DIR), 1)
    test_skch_imgs = load_images(get_file_list(TEST_SKCH_DIR), 0)

    test_hint_imgs, makeup = input_makeup(test_hint_imgs)
    test_skch_imgs, makeup = input_makeup(test_skch_imgs)

    for epoch in range(init_epoch, N_EPOCH):
        block_cnt = 0
        for skch_block_list, rgb_block_list in minibatches([skch_list, rgb_list], BLOCK_SIZE):
            block_cnt += 1
            # rgb_block, skch_block = load_clipped_images(rgb_block_list, skch_block_list, [1, 0])
            rgb_block = load_images(rgb_block_list, 1)
            skch_block = load_images(skch_block_list, 0)
            hint_block = get_hints(skch_block, rgb_block)
            hint_block = blur(hint_block)
            step_cnt = 0
            for skch_train, hint_train, rgb_train in minibatches([skch_block,\
                                        hint_block, rgb_block], BATCH_SIZE):
                step_start_time = time.time()
                step_cnt += 1
                feed_dict = {gan.rgb_imgs: rgb_train, gan.hint_imgs: hint_train, gan.skch_imgs: skch_train}
                d_loss, _ = sess.run([gan.d_loss, gan.d_op], feed_dict=feed_dict)
                g_loss, _ = sess.run([gan.g_loss, gan.g_op], feed_dict=feed_dict)
                log_content = '%d epoch, %d block counts, %d steps, d_loss: %.4f, g_loss: %.4f, step time: %.2f'\
                         % (epoch, block_cnt, step_cnt, d_loss, g_loss, time.time() - step_start_time)
                print(log_content)
                log('log_'+str(BATCH_SIZE), log_content+'\n')

            if (block_cnt + 1) % TEST_SAVING_PER_BLOCK == 0:
                save_tests(gan, sess, [test_skch_imgs, test_hint_imgs], TEST_RESULT_SAVING, 'latest', makeup)

        if (epoch + 1) % MODEL_SAVING_PER_EPOCH == 0:
            log('log_'+str(BATCH_SIZE), '%d Epoch Model Saving...\n' % (epoch))
            save_paths = [os.path.join(MODEL_SAVING, 'g_net_' + str(BATCH_SIZE) + '_' + str(epoch + 1) + '.npz'),\
                        os.path.join(MODEL_SAVING, 'd_net_' + str(BATCH_SIZE) + '_' + str(epoch + 1) + '.npz')]
            gan.save_model(save_paths, sess)
            log('log_'+str(BATCH_SIZE), 'Model Saved! \n')

        if (epoch + 1) % TEST_SAVING_PER_EPOCH == 0:
            save_tests(gan, sess, [test_skch_imgs, test_hint_imgs], TEST_RESULT_SAVING, str(epoch + 1), makeup)

def test(model_paths, save_dir, skch_dir, hint_dir, blurred = True):
    gan = model.Model(INPUT_SIZE, BATCH_SIZE, MEAN_ERROR_SCALING, LEARNING_RATE, ADAM_BETA1)
    sess = tf.InteractiveSession()
    gan.load_model(model_paths, sess)
    skches = load_images(get_file_list(skch_dir), 0)
    hints = load_images(get_file_list(hint_dir))
    skches, _ = input_makeup(skches)
    hints, makeup = input_makeup(hints)
    save_tests(gan, sess, [skches, hints], save_dir, 'test', makeup, blurred = blurred)

if __name__ == '__main__':
    train()
