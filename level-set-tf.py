import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import MeanIoU
from data import DataGenerator
from glob import glob
from network import ddunet
from utils import *
import pandas as pd
import logging
import matplotlib.pyplot as plt
from keras import backend as K

class ActiveContourModel:
    def __init__(self, config):
        # Configuration parameters
        self.config = config
        self.iter_limit = config['iter_limit']
        self.narrow_band_width = config['narrow_band_width']
        self.mu = config['mu']
        self.nu = config['nu']
        self.f_size = config['f_size']
        self.image_shape1 = config['image_shape1']
        self.image_shape2 = config['image_shape2']
        self.fast_lookup = config['fast_lookup']

        self.tumor_label = config['tumor_label']
        self.background_label = config['background_label']
        
    # This function draws the new contour based on the parameters derived previous to this step.
    # It gets the previous phi (contour), and the amount of changes (dt) we want the contour to have.
    # The output is the new phi (contour)
    def re_init_phi(self, phi, dt):
        D_left_shift = tf.cast(tf.roll(phi, -1, axis=1), dtype='float32')
        D_right_shift = tf.cast(tf.roll(phi, 1, axis=1), dtype='float32')
        D_up_shift = tf.cast(tf.roll(phi, -1, axis=0), dtype='float32')
        D_down_shift = tf.cast(tf.roll(phi, 1, axis=0), dtype='float32')
        bp = D_left_shift - phi
        cp = phi - D_down_shift
        dp = D_up_shift - phi
        ap = phi - D_right_shift
        an = tf.identity(ap)
        bn = tf.identity(bp)
        cn = tf.identity(cp)
        dn = tf.identity(dp)
        ap = tf.clip_by_value(ap, 0, 10 ** 38)
        bp = tf.clip_by_value(bp, 0, 10 ** 38)
        cp = tf.clip_by_value(cp, 0, 10 ** 38)
        dp = tf.clip_by_value(dp, 0, 10 ** 38)
        an = tf.clip_by_value(an, -10 ** 38, 0)
        bn = tf.clip_by_value(bn, -10 ** 38, 0)
        cn = tf.clip_by_value(cn, -10 ** 38, 0)
        dn = tf.clip_by_value(dn, -10 ** 38, 0)
        area_pos = tf.where(phi > 0)
        area_neg = tf.where(phi < 0)
        pos_y = area_pos[:, 0]
        pos_x = area_pos[:, 1]
        neg_y = area_neg[:, 0]
        neg_x = area_neg[:, 1]
        tmp1 = tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [ap, bn]], axis=0)
        tmp1 += tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [cp, dn]], axis=0)
        update1 = tf.sqrt(tf.abs(tmp1)) - 1
        indices1 = tf.stack([pos_y, pos_x], 1)
        tmp2 = tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [an, bp]], axis=0)
        tmp2 += tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [cn, dp]], axis=0)
        update2 = tf.sqrt(tf.abs(tmp2)) - 1
        indices2 = tf.stack([neg_y, neg_x], 1)
        indices_final = tf.concat([indices1, indices2], 0)
        update_final = tf.concat([update1, update2], 0)
        dD = tf.scatter_nd(indices_final, update_final, shape=[self.image_shape1, self.image_shape2])
        S = tf.divide(phi, tf.square(phi) + 1)
        phi = phi - tf.multiply(dt * S, dD)

        return phi
# This function gets phi (contour), x, andy, and calculates the curve of the contour
    def get_curvature(self, phi, x, y):
        phi_shape = tf.shape(phi)
        dim_x = phi_shape[1]
        dim_y = phi_shape[0]
        x = tf.cast(x, dtype="int32")
        y = tf.cast(y, dtype="int32")
        y_plus = tf.cast(y + 1, dtype="int32")
        y_minus = tf.cast(y - 1, dtype="int32")
        x_plus = tf.cast(x + 1, dtype="int32")
        x_minus = tf.cast(x - 1, dtype="int32")
        y_plus = tf.minimum(tf.cast(y_plus, dtype="int32"), tf.cast(dim_y - 1, dtype="int32"))
        x_plus = tf.minimum(tf.cast(x_plus, dtype="int32"), tf.cast(dim_x - 1, dtype="int32"))
        y_minus = tf.maximum(y_minus, 0)
        x_minus = tf.maximum(x_minus, 0)
        d_phi_dx = tf.gather_nd(phi, tf.stack([y, x_plus], 1)) - tf.gather_nd(phi, tf.stack([y, x_minus], 1))
        d_phi_dx_2 = tf.square(d_phi_dx)
        d_phi_dy = tf.gather_nd(phi, tf.stack([y_plus, x], 1)) - tf.gather_nd(phi, tf.stack([y_minus, x], 1))
        d_phi_dy_2 = tf.square(d_phi_dy)
        d_phi_dxx = tf.gather_nd(phi, tf.stack([y, x_plus], 1)) + tf.gather_nd(phi, tf.stack([y, x_minus], 1)) - \
                    2 * tf.gather_nd(phi, tf.stack([y, x], 1))
        d_phi_dyy = tf.gather_nd(phi, tf.stack([y_plus, x], 1)) + tf.gather_nd(phi, tf.stack([y_minus, x], 1)) - \
                    2 * tf.gather_nd(phi, tf.stack([y, x], 1))
        d_phi_dxy = 0.25 * (- tf.gather_nd(phi, tf.stack([y_minus, x_minus], 1)) - tf.gather_nd(phi, tf.stack(
            [y_plus, x_plus], 1)) + tf.gather_nd(phi, tf.stack([y_minus, x_plus], 1)) + tf.gather_nd(phi, tf.stack(
            [y_plus, x_minus], 1)))
        tmp_1 = tf.multiply(d_phi_dx_2, d_phi_dyy) + tf.multiply(d_phi_dy_2, d_phi_dxx) - \
                2 * tf.multiply(tf.multiply(d_phi_dx, d_phi_dy), d_phi_dxy)
        tmp_2 = tf.add(tf.pow(d_phi_dx_2 + d_phi_dy_2, 1.5), 2.220446049250313e-16)
        tmp_3 = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)
        tmp_4 = tf.divide(tmp_1, tmp_2)
        curvature = tf.multiply(tmp_3, tmp_4)
        mean_grad = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)

        return curvature, mean_grad

    # This function gets the images, and the contour, and calculated the intesnity of the image inside of the contour
    def get_intensity(self, image, masked_phi, filter_patch_size=5, filter_depth_size=1):
        b_exp = image
        uu = tf.multiply(b_exp, masked_phi)
        u_1 = tf.keras.layers.AveragePooling2D([filter_patch_size, filter_patch_size], 1, padding='same')(uu)
        u_2 = tf.keras.layers.AveragePooling2D([filter_patch_size, filter_patch_size], 1, padding='same')(masked_phi)

        u_2_prime = 1 - tf.cast((u_2 > 0), dtype='float32') + tf.cast((u_2 < 0), dtype='float32')
        u_2 = u_2 + u_2_prime + 2.220446049250313e-16

        return tf.divide(u_1, u_2)

    # This function gets the image, initial contour (phi), and the mapped lambdas as input 
    def active_contour_layer(self, elems):
        img = elems[0]
        init_phi = elems[1]
        map_lambda1_acl = elems[2]
        map_lambda2_acl = elems[3]
        wind_coef = 3
        zero_tensor = tf.constant(0, shape=[], dtype="int32")
        
        def _body(i, phi_level):
            band_index = tf.reduce_all([phi_level <= self.narrow_band_width, phi_level >= -self.narrow_band_width], axis=0)
            band = tf.where(band_index)
            band_y = band[:, 0]
            band_x = band[:, 1]
            shape_y = tf.shape(band_y)
            num_band_pixel = shape_y[0]
            window_radii_x = tf.ones(num_band_pixel) * wind_coef
            window_radii_y = tf.ones(num_band_pixel) * wind_coef

            def body_intensity(j, mean_intensities_outer, mean_intensities_inner):
                xnew = tf.cast(band_x[j], dtype="float32")
                ynew = tf.cast(band_y[j], dtype="float32")
                window_radius_x = tf.cast(window_radii_x[j], dtype="int32")
                window_radius_y = tf.cast(window_radii_y[j], dtype="int32")
                window_x = tf.range(tf.maximum(tf.cast(xnew - window_radius_x, dtype="int32"), zero_tensor),
                                    tf.minimum(tf.cast(xnew + window_radius_x + 1, dtype="int32"),
                                               tf.shape(img)[1]))
                window_y = tf.range(tf.maximum(tf.cast(ynew - window_radius_y, dtype="int32"), zero_tensor),
                                    tf.minimum(tf.cast(ynew + window_radius_y + 1, dtype="int32"),
                                               tf.shape(img)[0]))

                window_pixels = tf.stack(tf.meshgrid(window_y, window_x, indexing='ij'), axis=-1)
                intensity = self.get_intensity(img, tf.gather_nd(init_phi, window_pixels))
                mean_intensities_outer = tf.tensor_scatter_nd_update(mean_intensities_outer, [[j]], [tf.reduce_mean(intensity)])
                mean_intensities_inner = tf.tensor_scatter_nd_update(mean_intensities_inner, [[j]], [self.get_intensity(img, tf.gather_nd(init_phi, window_pixels))])

                return j + 1, mean_intensities_outer, mean_intensities_inner
# Here we have two option of calculating the contour at each iteration, either go througha fast look-up, or do the slower calculations.
# They both have the same function, but the fast_lookup is less deatiled and computationally faster
            if fast_lookup:
                phi_4d = phi_level[tf.newaxis, :, :, tf.newaxis]
                image = img[tf.newaxis, :, :, tf.newaxis]
                band_index_2 = tf.reduce_all([phi_4d <= narrow_band_width, phi_4d >= -narrow_band_width], axis=0)
                band_2 = tf.where(band_index_2)
                u_inner = get_intensity(image, tf.cast((([phi_4d <= 0])), dtype='float32')[0], filter_patch_size=f_size)
                u_outer = get_intensity(image, tf.cast((([phi_4d > 0])), dtype='float32')[0], filter_patch_size=f_size)
                mean_intensities_inner = tf.gather_nd(u_inner, band_2)
                mean_intensities_outer = tf.gather_nd(u_outer, band_2)

            else:
                mean_intensities_inner = tf.constant([0], dtype='float32')
                mean_intensities_outer = tf.constant([0], dtype='float32')
                j = tf.constant(0, dtype=tf.int32) 
                _, mean_intensities_outer, mean_intensities_inner = tf.while_loop(
                    lambda j, mean_intensities_outer, mean_intensities_inner:
                    j < num_band_pixel, body_intensity, loop_vars=[j, mean_intensities_outer, mean_intensities_inner],
                    shape_invariants=[j.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])

            lambda1 = tf.gather_nd(map_lambda1_acl, [band])
            lambda2 = tf.gather_nd(map_lambda2_acl, [band])
            curvature, mean_grad = get_curvature(phi_level, band_x, band_y)
            kappa = tf.multiply(curvature, mean_grad)
            term1 = tf.multiply(tf.cast(lambda1, dtype='float32'),tf.square(tf.gather_nd(img, [band]) - mean_intensities_inner))
            term2 = tf.multiply(tf.cast(lambda2, dtype='float32'),tf.square(tf.gather_nd(img, [band]) - mean_intensities_outer))
            force = -nu + term1 - 10*term2
            force /= (tf.reduce_max(tf.abs(force)))
            d_phi_dt = tf.cast(force, dtype="float32") + tf.cast(mu * kappa, dtype="float32")
            dt = .45 / (tf.reduce_max(tf.abs(d_phi_dt)) + 2.220446049250313e-16)
            d_phi = dt * d_phi_dt
            update_narrow_band = d_phi
            phi_level = phi_level + tf.scatter_nd([band], tf.cast(update_narrow_band, dtype='float32'),shape=[image_shape1, image_shape2])
            phi_level = re_init_phi(phi_level, 0.5)
            
            return i + 1, phi_level

            _, mean_intensities_outer, mean_intensities_inner = tf.while_loop(
                lambda j, *_: j < num_band_pixel,
                body_intensity,
                [zero_tensor, tf.zeros(num_band_pixel), tf.zeros(num_band_pixel)]
            )

            mean_inner = tf.reduce_mean(mean_intensities_inner)
            mean_outer = tf.reduce_mean(mean_intensities_outer)
            dist_transform = tf.sqrt(tf.reduce_sum(tf.square(mean_intensities_outer - mean_intensities_inner), axis=0))
            phi_level = phi_level - tf.multiply(self.mu, dist_transform - (mean_inner - mean_outer))

            return i + 1, phi_level
            
        i = tf.constant(0, dtype=tf.int32)
        phi = init_phi
        print(iter_limit, 'aaaa')
        _, phi = tf.while_loop(lambda i, phi: i < iter_limit, _body, loop_vars=[i, phi])
        phi = tf.round(tf.cast((1 - tf.nn.sigmoid(phi)), dtype=tf.float32))
        
        return phi,init_phi, map_lambda1_acl, map_lambda2_acl

        # _, phi_result = tf.while_loop(
        #     lambda i, _: i < self.iter_limit,
        #     _body,
        #     [tf.constant(0), init_phi]
        # )

        # return phi_result

    # def train(self):
    #     # Implement your training logic here
    #     pass

def parse_args():
    parser = argparse.ArgumentParser(description='Active Contour Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    return parser.parse_args()

def main():
    args = parse_args()
    config = read_config(args.config)
    model = ActiveContourModel(config)
    # model.train()
    batch_size = 1
    train_sum_freq = 150
    img_resize = 128
    f_size = 15
    train_status = 1
    narrow_band_width = 3
    save_freq = 1000
    demo_type = 1
    gpu = 0
    gpu_id = 1
    # The paths to the image, masks, and the initial contour.
    all_path_input = "/workspace/chase/image/*.jpg"
    all_path_mask = "/workspace/chase/mask/*.png"
    all_path_seg = "/workspace/chase/image/*_acm.npy"
    
    img_paths = glob(all_path_input)
    all_mask_paths = glob(all_path_mask)
    seg_paths = glob(all_path_seg)
    
    img_paths.sort()
    all_mask_paths.sort()
    seg_paths.sort()
    # Here we are intializing the mu, nu, and the number of iterations for the acm.
    # mu is a parameter controlling the effect of penalizing the deviation of from a signed distance
    # nu nfluences the smoothness term in the energy functional. This term penalizes irregularities in the contour, promoting smoother and more regular shapes
    # iter_num is the number of iterations.
    # Higher nu : Increases the influence of the smoothness term, leading to a smoother contour. This can help in avoiding overfitting to noise or small variations in the image but might lead to less accurate segmentation in cases where the actual contour is not smooth.
    # Lower nu: Reduces the penalty for irregularities in the contour, allowing the contour to fit more closely to the image features. However, this may lead to less smooth contours and potential overfitting to noise or artifacts in the image.
    SMOOTH = 1e-6
    mu = 0.1
    nu = 2
    iter_limit = 600
    for i in range(len(img_paths)):
        print('Processing Case {} '.format(i+1), img_paths[i], iter_limit)
        id = img_paths[i].split('.')[0]
        labels = load_image(all_mask_paths[i], 1, True)
        image = load_image(img_paths[i], 1, False)
        image = (image/image.max())
        image_shape1 = image.shape[1]
        image_shape2 = image.shape[2]
        out_seg = load_image(seg_paths[i], 1, True)
        gt_mask = labels
    
        x_acm = image
        map_lambda1 = tf.exp(tf.divide(tf.subtract(2.0,out_seg),tf.add(1.0,out_seg)))
        map_lambda2 = tf.exp(tf.divide(tf.add(1.0, out_seg), tf.subtract(2.0, out_seg)))
        y_out_dl = tf.round(out_seg)
        rounded_seg_acl = y_out_dl
        dt_trans = tf.py_function(my_func, [rounded_seg_acl], tf.float32)
        dt_trans.set_shape([batch_size, image_shape1, image_shape2])
        seg_out_acm, _, lambda1_tr, lambda2_tr = tf.map_fn(fn=model.active_contour_layer(), elems=(x_acm, dt_trans, map_lambda1, map_lambda2))
    
        fig, axs = plt.subplots(1, 4, figsize=(9, 3))
        axs[0].imshow(image[0, :, :])
        axs[1].imshow(labels[0, :, :], cmap='gray')
        axs[2].imshow(out_seg[0, :, :], cmap='gray')
        axs[3].imshow(seg_out_acm[0, :, :], cmap='gray')
        plt.show()

if __name__ == '__main__':
    main()
