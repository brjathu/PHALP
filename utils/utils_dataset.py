import torch
import numpy as np
import random
import cv2
# from easydict import EasyDict as edict


def process_image(img, center, scale):
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])

    img, _, _ = generate_image_patch(img, center[0], center[1], scale, scale, 256, 256, False, 1.0, 0.0)
    img = img[:, :, ::-1].copy().astype(np.float32)
    img_n = img[:, :, ::-1].copy().astype(np.float32)
    for n_c in range(3):
        img_n[:, :, n_c] = (img_n[:, :, n_c] - mean[n_c]) / std[n_c]
    return torch.from_numpy(np.transpose(img_n, (2, 0, 1)))


def process_mask(img, center, scale):
    img, _, _ = generate_image_patch(img, center[0], center[1], scale, scale, 256, 256, False, 1.0, 0.0)
    img_n = img[:, :, ::-1].copy().astype(np.float32)
    return torch.from_numpy(np.transpose(img_n, (2, 0, 1)))

# def process_mask2(img, center, scale):
#     img, _, _ = generate_image_patch(img, center[0], center[1], scale, scale, 256, 256, False, 1.0, 0.0)
#     # img_n = img[:, :, ::-1].copy().astype(np.float32)
#     return img # torch.from_numpy(np.transpose(img_n, (2, 0, 1)))


# def process_image_wh(img, center, s1, s2, w, h):
#     mean = np.array([123.675, 116.280, 103.530])
#     std = np.array([58.395, 57.120, 57.375])

#     img, _, _ = generate_image_patch(img, center[0], center[1], s1, s2, w, h, False, 1.0, 0.0)
#     img = img[:, :, ::-1].copy().astype(np.float32)
#     img_n = img[:, :, ::-1].copy().astype(np.float32)
#     for n_c in range(3):
#         img_n[:, :, n_c] = (img_n[:, :, n_c] - mean[n_c]) / std[n_c]
#     return torch.from_numpy(np.transpose(img_n, (2, 0, 1)))




# def generate_image_patch_mask(mask, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
#     mask_2 = mask.copy()
#     # c = center.copy()
#     img_height, img_width = mask_2.shape

#     if do_flip:
#         mask_2 = mask_2[:, ::-1]
#         c_x = img_width - c_x - 1

#     trans, trans_inv = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)
    
#     mask_2 = np.float32(mask_2)
    
#     img_patch = cv2.warpAffine(mask_2, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

#     return img_patch, trans, trans_inv



# def get_default_augment_config():
#     config = edict()
#     config.scale_factor = 0.2
#     config.rot_factor = 30
#     config.color_factor = 0.2
#     config.do_flip_aug = True

#     config.rot_aug_rate = 0.6  #possibility to rot aug
#     config.flip_aug_rate = 0.5 #possibility to flip aug
#     return config


# def do_augmentation():
#     aug_config = get_default_augment_config()

#     tx = np.clip(np.random.randn(), -1.0, 1.0) * 0.02
#     ty = np.clip(np.random.randn(), -1.0, 1.0) * 0.02
#     scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.scale_factor + 1.1
#     rot = np.clip(np.random.randn(), -2.0,
#                   2.0) * aug_config.rot_factor if random.random() <= aug_config.rot_aug_rate else 0
#     do_flip = aug_config.do_flip_aug and random.random() <= aug_config.flip_aug_rate
#     c_up = 1.0 + aug_config.color_factor
#     c_low = 1.0 - aug_config.color_factor
#     color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
#     return scale, rot, do_flip, color_scale, tx, ty

# def get_bbox(keypoints, rescale=1.2, detection_thresh=0.0):
#     """Get center and scale for bounding box from openpose detections."""
#     valid = keypoints[:,-1] > detection_thresh
#     valid_keypoints = keypoints[valid][:,:-1]
#     center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
#     bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
#     # adjust bounding box tightness
#     scale = bbox_size
#     scale *= rescale
#     return center, scale

# def crop_to_hips(center_x, center_y, width, height, keypoints_2d):
#     keypoints_2d = keypoints_2d.copy()
#     lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24, -19, -18, -15, -14]
#     keypoints_2d[lower_body_keypoints, :] = 0
#     if keypoints_2d[:, -1].sum() > 1:
#         center, scale = get_bbox(keypoints_2d)
#         center_x = center[0]
#         center_y = center[1]
#         width = 1.1 * scale[0]
#         height = 1.1 * scale[1]
#     return center_x, center_y, width, height

# def crop_to_shoulders(center_x, center_y, width, height, keypoints_2d):
#     keypoints_2d = keypoints_2d.copy()
#     lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, -19, -18, -17, -16, -15, -14, -13, -12, -9, -8, -5, -3]
#     keypoints_2d[lower_body_keypoints, :] = 0
#     keypoints_2d[25:25+2*21] = 0.
#     center, scale = get_bbox(keypoints_2d)
#     if keypoints_2d[:, -1].sum() > 1:
#         center, scale = get_bbox(keypoints_2d)
#         center_x = center[0]
#         center_y = center[1]
#         width = 1.2 * scale[0]
#         height = 1.2 * scale[1]
#     return center_x, center_y, width, height

# def crop_to_head(center_x, center_y, width, height, keypoints_2d):
#     keypoints_2d = keypoints_2d.copy()
#     lower_body_keypoints = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -5, -3]
#     keypoints_2d[lower_body_keypoints, :] = 0
#     keypoints_2d[25:25+2*21] = 0.
#     if keypoints_2d[:, -1].sum() > 1:
#         center, scale = get_bbox(keypoints_2d)
#         center_x = center[0]
#         center_y = center[1]
#         width = 1.3 * scale[0]
#         height = 1.3 * scale[1]
#     return center_x, center_y, width, height

# def full_body(keypoints_2d):
#     body_keypoints = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14, -17, -16]
#     return (keypoints_2d[body_keypoints, -1] > 0).sum() == len(body_keypoints)

# def upper_body(keypoints_2d):
#     lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
#     upper_body_keypoints = [0, 1, 2, 5, -7, -6]
#     return ((keypoints_2d[lower_body_keypoints, -1] > 0).sum() == 0) and ((keypoints_2d[upper_body_keypoints, -1] > 0).sum() >= 2)

# def extreme_cropping(center_x, center_y, width, height, keypoints_2d):
#     p = torch.rand(1).item()
#     if full_body(keypoints_2d):
#         if p < 0.7:
#             center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
#         elif p < 0.9:
#             center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
#         else:
#             center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
#     elif upper_body(keypoints_2d):
#         if p < 0.9:
#             center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
#         else:
#             center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
#     return center_x, center_y, max(width, height), max(width, height)

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv


# def trans_point2d(pt_2d, trans):
#     src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
#     dst_pt = np.dot(trans, src_pt)
#     return dst_pt[0:2]


def generate_image_patch(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    # c = center.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans, trans_inv = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return img_patch, trans, trans_inv




# def convert_cvimg_to_tensor(cvimg, occlusion_aug=True):
#     # from h,w,c(OpenCV) to c,h,w
#     tensor = cvimg.copy()
#     tensor = np.transpose(tensor, (2, 0, 1))
#     # from BGR(OpenCV) to RGB
#     # tensor = tensor[::-1, :, :]
#     # from int to float
#     tensor = tensor.astype(np.float32)
#     return tensor

# def swap(x, y):
#     tmp = x.copy()
#     x = y
#     y = tmp
#     return x, y

# def fliplr_params(smplx_params, has_smplx_params, body_pose_permutation=None):
#     global_orient = smplx_params['global_orient'].copy()
#     body_pose = smplx_params['body_pose'].copy()
#     left_hand_pose = smplx_params['left_hand_pose'].copy()
#     right_hand_pose = smplx_params['right_hand_pose'].copy()
#     leye_pose = smplx_params['leye_pose'].copy()
#     reye_pose = smplx_params['reye_pose'].copy()
#     jaw_pose = smplx_params['jaw_pose'].copy()
#     betas = smplx_params['betas'].copy()
#     expression = smplx_params['expression'].copy()

#     has_global_orient = has_smplx_params['global_orient'].copy()
#     has_body_pose = has_smplx_params['body_pose'].copy()
#     has_left_hand_pose = has_smplx_params['left_hand_pose'].copy()
#     has_right_hand_pose = has_smplx_params['right_hand_pose'].copy()
#     has_leye_pose = has_smplx_params['leye_pose'].copy()
#     has_reye_pose = has_smplx_params['reye_pose'].copy()
#     has_jaw_pose = has_smplx_params['jaw_pose'].copy()
#     has_betas = has_smplx_params['betas'].copy()
#     has_expression = has_smplx_params['expression'].copy()

#     body_pose_permutation = [6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13,
#                              14 ,18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33,
#                              34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41,
#                              45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55,
#                              56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
#     body_pose_permutation = body_pose_permutation[:len(body_pose)]
#     body_pose_permutation = [i-3 for i in body_pose_permutation]

#     body_pose = body_pose[body_pose_permutation]
#     left_hand_pose, right_hand_pose = swap(left_hand_pose, right_hand_pose)
#     leye_pose, reye_pose = swap(leye_pose, reye_pose)
#     has_left_hand_pose, has_right_hand_pose = swap(has_left_hand_pose, has_right_hand_pose)
#     has_leye_pose, has_reye_pose = swap(has_leye_pose, has_reye_pose)
#     has_expression *= 0

#     global_orient[1::3] *= -1
#     global_orient[2::3] *= -1
#     jaw_pose[1::3] *= -1
#     jaw_pose[2::3] *= -1
#     leye_pose[1::3] *= -1
#     leye_pose[2::3] *= -1
#     reye_pose[1::3] *= -1
#     reye_pose[2::3] *= -1
#     body_pose[1::3] *= -1
#     body_pose[2::3] *= -1

#     smplx_params = {'global_orient': global_orient.astype(np.float32),
#                     'body_pose': body_pose.astype(np.float32),
#                     'left_hand_pose': left_hand_pose.astype(np.float32),
#                     'right_hand_pose': right_hand_pose.astype(np.float32),
#                     'leye_pose': leye_pose.astype(np.float32),
#                     'reye_pose': reye_pose.astype(np.float32),
#                     'jaw_pose': jaw_pose.astype(np.float32),
#                     'betas': betas.astype(np.float32),
#                     'expression': expression.astype(np.float32),
#                    }

#     has_smplx_params = {'global_orient': has_global_orient,
#                         'body_pose': has_body_pose,
#                         'left_hand_pose': has_left_hand_pose,
#                         'right_hand_pose': has_right_hand_pose,
#                         'leye_pose': has_leye_pose,
#                         'reye_pose': has_reye_pose,
#                         'jaw_pose': has_jaw_pose,
#                         'betas': has_betas,
#                         'expression': has_expression,
#                        }

#     return smplx_params, has_smplx_params


# def fliplr_keypoints(joints, width, flip_permutation):
#     joints = joints.copy()
#     # Flip horizontal
#     joints[:, 0] = width - joints[:, 0] - 1

#     joints = joints[flip_permutation, :]

#     return joints

# def keypoint_3d_processing(S, flip_permutation, r, f):
#     if f:
#         S = fliplr_keypoints(S, 1, flip_permutation)
#     # in-plane rotation
#     rot_mat = np.eye(3)
#     if not r == 0:
#         rot_rad = -r * np.pi / 180
#         sn,cs = np.sin(rot_rad), np.cos(rot_rad)
#         rot_mat[0,:2] = [cs, -sn]
#         rot_mat[1,:2] = [sn, cs]
#     S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
#     # flip the x coordinates
#     S = S.astype('float32')
#     return S

# def rot_aa(aa, rot):
#     """Rotate axis angle parameters."""
#     # pose parameters
#     R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
#                   [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
#                   [0, 0, 1]])
#     # find the rotation of the body in camera frame
#     per_rdg, _ = cv2.Rodrigues(aa)
#     # apply the global rotation to the global orientation
#     resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
#     aa = (resrot.T)[0]
#     return aa.astype(np.float32)

# def smplx_param_processing(smplx_params, has_smplx_params, rot, flip):
#     if flip:
#         smplx_params, has_smplx_params = fliplr_params(smplx_params, has_smplx_params)
#     smplx_params['global_orient'] = rot_aa(smplx_params['global_orient'], rot)
#     return smplx_params, has_smplx_params



# def get_example(img_path, center_x, center_y, width, height,
#                 keypoints_2d, keypoints_3d,
#                 smplx_params, has_smplx_params,
#                 flip_kp_permutation,
#                 patch_width, patch_height,
#                 mean, std, do_augment, excrop):
#     # 1. load image
#     cvimg = cv2.imread( img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
#     if not isinstance(cvimg, np.ndarray):
#         raise IOError("Fail to read %s" % img_path)

#     img_height, img_width, img_channels = cvimg.shape
    
    
#     mask_ = np.load("/home/jathu/MultiObjectTracking/All/Masks/" + img_path.split("_Datasets")[1][:-4] + ".npy")
#     if(len(mask_.shape)>1):
#         scores_ = np.zeros((mask_.shape[0],))
#         for i in range(mask_.shape[0]):
#             x_ = np.where(np.sum(mask_[i], 0)>=1)
#             x_0 = np.min(x_)
#             x_1 = np.max(x_)

#             y_ = np.where(np.sum(mask_[i], 1)>=1)
#             y_0 = np.min(y_)
#             y_1 = np.max(y_)


#             scores_[i] =  abs( 0.5*(x_0 + x_1) - center_x )  +\
#                           abs( 0.5*(y_0 + y_1) - center_y )  +\
#                           abs( (y_1 - y_0) - width )
#         mask = mask_[np.argmin(scores_)]
#     else:
#         mask = np.zeros((img_height, img_width), np.float32) + 1.0
        
    
    
    
#     # 2. get augmentation params
#     if do_augment:
#         scale, rot, do_flip, color_scale, tx, ty = do_augmentation()
#         # sparse extreme cropping
#         if excrop and torch.rand(1).item() > 0.9:
#             center_x, center_y, width, height = extreme_cropping(center_x, center_y, width, height, keypoints_2d)
#     else:
#         scale, rot, do_flip, color_scale, tx, ty = 1.0, 0, False, [1.0, 1.0, 1.0], 0., 0.

#     center_x += width * tx
#     center_y += height * ty

#     # Process 3D keypoints
#     keypoints_3d = keypoint_3d_processing(keypoints_3d, flip_kp_permutation, rot, do_flip)

#     # 3. generate image patch
#     img_patch_cv, trans, trans_inv = generate_image_patch(cvimg,
#                                                           center_x, center_y,
#                                                           width, height,
#                                                           patch_width, patch_height,
#                                                           do_flip, scale, rot)
    
    
#     image = img_patch_cv.copy()
#     image = image[:, :, ::-1]

#     mask_patch_cv, _, _ = generate_image_patch_mask(mask,
#                                                           center_x, center_y,
#                                                           width, height,
#                                                           patch_width, patch_height,
#                                                           do_flip, scale, rot)
    

#     img_patch_cv = image.copy()
#     img_patch = convert_cvimg_to_tensor(image)

#     smplx_params, has_smplx_params = smplx_param_processing(smplx_params, has_smplx_params, rot, do_flip)

#     # apply normalization
#     for n_c in range(img_channels):
#         img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
#         if mean is not None and std is not None:
#             img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
#     if do_flip:
#         keypoints_2d = fliplr_keypoints(keypoints_2d, img_width, flip_kp_permutation)


#     for n_jt in range(len(keypoints_2d)):
#         keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
#     keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5

#     return img_patch, keypoints_2d, keypoints_3d, smplx_params, has_smplx_params, mask_patch_cv


# def get_example_noimg(center_x, center_y, width, height,
#                 keypoints_2d, keypoints_3d,
#                 smplx_params, has_smplx_params,
#                 flip_kp_permutation,
#                 patch_width, patch_height,
#                 mean, std, do_augment, excrop):

#     # 2. get augmentation params
#     if do_augment:
#         scale, rot, do_flip, color_scale, tx, ty = do_augmentation()
#         if excrop and torch.rand(1).item() > 0.9:
#             center_x, center_y, width, height = extreme_cropping(center_x, center_y, width, height, keypoints_2d)
#     else:
#         scale, rot, do_flip, color_scale, tx, ty = 1.0, 0, False, [1.0, 1.0, 1.0], 0., 0.

#     center_x += width * tx
#     center_y += height * ty

#     # Process 3D keypoints
#     keypoints_3d = keypoint_3d_processing(keypoints_3d, flip_kp_permutation, rot, do_flip)

#     # 3. generate image patch
#     trans, trans_inv = gen_trans_from_patch_cv(center_x, center_y, width, height, patch_width, patch_height, scale, rot, inv=False)

#     smplx_params, has_smplx_params = smplx_param_processing(smplx_params, has_smplx_params, rot, do_flip)

#     for n_jt in range(len(keypoints_2d)):
#         keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
#     keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5

#     return keypoints_2d, keypoints_3d, smplx_params, has_smplx_params
