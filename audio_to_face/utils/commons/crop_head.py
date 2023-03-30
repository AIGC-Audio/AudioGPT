import face_alignment
import os
import cv2
import skimage.transform as trans
import argparse
import torch
import numpy as np
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_affine(src):
    dst = np.array([[87,  59],
                    [137,  59],
                    [112, 120]], dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    return M


def affine_align_img(img, M, crop_size=224):
    warped = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
    return warped


def affine_align_3landmarks(landmarks, M):
    new_landmarks = np.concatenate([landmarks, np.ones((3, 1))], 1)
    affined_landmarks = np.matmul(new_landmarks, M.transpose())
    return affined_landmarks


def get_eyes_mouths(landmark):
    three_points = np.zeros((3, 2))
    three_points[0] = landmark[36:42].mean(0)
    three_points[1] = landmark[42:48].mean(0)
    three_points[2] = landmark[60:68].mean(0)
    return three_points


def get_mouth_bias(three_points):
    bias = np.array([112, 120]) - three_points[2]
    return bias


def align_folder(folder_path, folder_save_path):

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
    preds = fa.get_landmarks_from_directory(folder_path)

    sumpoints = 0
    three_points_list = []

    for img in tqdm.tqdm(preds.keys(), desc='preprocessing..'):
        pred_points = np.array(preds[img])
        if pred_points is None or len(pred_points.shape) != 3:
            print('preprocessing failed')
            return False
        else:
            num_faces, size, _ = pred_points.shape
            if num_faces == 1 and size == 68:

                three_points = get_eyes_mouths(pred_points[0])
                sumpoints += three_points
                three_points_list.append(three_points)
            else:

                print('preprocessing failed')
                return False
    avg_points = sumpoints / len(preds)
    M = get_affine(avg_points)
    p_bias = None
    for i, img_pth in tqdm.tqdm(enumerate(preds.keys()), desc='affine and save'):
        three_points = three_points_list[i]
        affined_3landmarks = affine_align_3landmarks(three_points, M)
        bias = get_mouth_bias(affined_3landmarks)
        if p_bias is None:
            bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M_i = M.copy()
        M_i[:, 2] = M[:, 2] + bias
        img = cv2.imread(img_pth)
        wrapped = affine_align_img(img, M_i)
        img_save_path = os.path.join(folder_save_path, img_pth.split('/')[-1])
        cv2.imwrite(img_save_path, wrapped)
    print('cropped files saved at {}'.format(folder_save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', help='the folder which needs processing')
    args = parser.parse_args()

    if os.path.isdir(args.folder_path):
        home_path = '/'.join(args.folder_path.split('/')[:-1])
        save_img_path = os.path.join(home_path, args.folder_path.split('/')[-1] + '_cropped')
        os.makedirs(save_img_path, exist_ok=True)

        align_folder(args.folder_path, save_img_path)


if __name__ == '__main__':
    main()
