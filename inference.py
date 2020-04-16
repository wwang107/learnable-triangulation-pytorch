import torch
import numpy as np
import cv2
import os
from collections import defaultdict 
from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss
from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from mvn.utils.multiview import Camera
from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion as project
from mvn.datasets import utils as dataset_utils


class Detector:
    def __init__(self, config, device = "cuda:0"):
        super().__init__()
        
        self.model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
        }[config.model.name](config, device=device).to(device)

        if config.model.init_weights:
            state_dict = torch.load(config.model.checkpoint)
            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)

            state_dict = torch.load(config.model.checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded pretrained weights for whole model")
        
    def inference(self, image, projMatrix,isSaveOutput=False, savePath=""):
        """
        For a single image inference
        """


        
        return

    def inferHuman36Data(self, batch, randomize_n_views,
                                        min_n_views,
                                        max_n_views):
        """
        For batch inferences 
        """
        outputBatch = {}
        inputBatch = {}
        collatFunction = dataset_utils.make_collate_fn(randomize_n_views,
                                        min_n_views,
                                        max_n_views)
        batch = collatFunction(batch)
        images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch  = dataset_utils.prepare_batch(batch, device, config)
        keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = self.model(images_batch, proj_matricies_batch, batch)

        outputBatch["keypoints_3d_pred"] = keypoints_3d_pred
        outputBatch["heatmaps_pred"] = heatmaps_pred
        outputBatch["volumes_pred"] = volumes_pred
        outputBatch["confidences_pred"] = confidences_pred
        outputBatch["cuboids_pred"] = confidences_pred
        outputBatch["coord_volumes_pred"] = coord_volumes_pred
        outputBatch["base_points_pred"] = base_points_pred

        inputBatch["images_batch"] = images_batch
        inputBatch["proj_matricies_batch"] = proj_matricies_batch
        return outputBatch, inputBatch

def viewSample(sample):
    camera_idx = 0
    image = sample['images'][camera_idx]
    camera = sample['cameras'][camera_idx]
    subject = sample['subject'][camera_idx]
    action = sample['action'][camera_idx]

    display = image.copy()
    keypoints_2d = project(camera.projection, sample['keypoints_3d'][:, :3])
    for i,(x,y) in enumerate(keypoints_2d):
        cv2.circle(display, (int(x), int(y)), 3, (0,0,255), -1)
    file = f"/home/weiwang/Desktop/{subject}-{action}-{camera.name}.png"

     

def prepareSample(idx, labels, human36mRoot, keyPoint3d = None , imageShape = None, scaleBox = 1.0, crop = True, normImage = False):
    sample = defaultdict(list) # return value
    shot = labels['table'][idx]
    subject = labels['subject_names'][shot['subject_idx']]
    action = labels['action_names'][shot['action_idx']]
    frame_idx = shot['frame_idx']

    for camera_idx, camera_name in enumerate(labels['camera_names']):
        bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
        bbox_height = bbox[2] - bbox[0]

        if bbox_height == 0:
            # convention: if the bbox is empty, then this view is missing
            continue

        # scale the bounding box
        bbox = scale_bbox(bbox, scaleBox)

        # load image
        image_path = os.path.join(human36mRoot, subject, action, 'imageSequence', camera_name, 'img_%06d.jpg' % (frame_idx+1))
        assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
        image = cv2.imread(image_path)
        
        # load camera
        shot_camera = labels['cameras'][shot['subject_idx'], camera_idx]
        retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

        if crop:
                # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)
                

        if imageShape is not None:
            # resize
            image_shape_before_resize = image.shape[:2]
            image = resize_image(image, imageShape)
            retval_camera.update_after_resize(image_shape_before_resize, imageShape)

            sample['image_shapes_before_resize'].append(image_shape_before_resize)

        if normImage:
            image = normalize_image(image)

        sample['images'].append(image)
        sample['detections'].append(bbox + (1.0,)) # TODO add real confidences
        sample['cameras'].append(retval_camera)
        sample['proj_matrices'].append(retval_camera.projection)
        sample["action"].append(action)
        sample["subject"].append(subject)
        sample["frameId"].append(frame_idx)
        # 3D keypoints
        # add dummy confidences
        sample['keypoints_3d'] = np.pad(
            shot['keypoints'][:17],
            ((0,0), (0,1)), 'constant', constant_values=1.0)

    # build cuboid
    # base_point = sample['keypoints_3d'][6, :3]
    # sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
    # position = base_point - sides / 2
    # sample['cuboids'] = volumetric.Cuboid3D(position, sides)

    # save sample's index
    sample['indexes'] = idx

    if keyPoint3d is not None:
        sample['pred_keypoints_3d'] = keyPoint3d[idx]

    sample.default_factory = None
    return sample

def loadHuman36mLabel(path,train = True, withDamageAction=True, retain_every_n_frames_in_test=1):
    """
    this load the label, including bouding box, camera matrices
    """
    test = not train
    labels = np.load(path, allow_pickle=True).item()
    train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    test_subjects = ['S9', 'S11']

    train_subjects = list(labels['subject_names'].index(x) for x in train_subjects)
    test_subjects  = list(labels['subject_names'].index(x) for x in test_subjects)

    indices = []
    if train:
        mask = np.isin(labels['table']['subject_idx'], train_subjects, assume_unique=True)
        indices.append(np.nonzero(mask)[0])
    if test:
        mask = np.isin(labels['table']['subject_idx'], test_subjects, assume_unique=True)

        if not withDamageAction:
            mask_S9 = labels['table']['subject_idx'] == labels['subject_names'].index('S9')

            damaged_actions = 'Greeting-2', 'SittingDown-2', 'Waiting-1'
            damaged_actions = [labels['action_names'].index(x) for x in damaged_actions]
            mask_damaged_actions = np.isin(labels['table']['action_idx'], damaged_actions)

            mask &= ~(mask_S9 & mask_damaged_actions)
        
            
        indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])
    labels['table'] = labels['table'][np.concatenate(indices)]
    return labels
    
    

def loadPrePelvis(path):
    pred_results = np.load(path, allow_pickle=True)
    keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
    return keypoints_3d_pred

if __name__ == "__main__":
    config = cfg.load_config("/home/weiwang/Desktop/master-thesis/learnable-triangulation-pytorch/experiments/human36m/train/human36m_vol_softmax.yaml")
    device = torch.device(0)
    labels = loadHuman36mLabel(config.dataset.train.labels_path)
    pelvis3d = loadPrePelvis(config.dataset.train.pred_results_path)
    sample = [prepareSample(100, labels, config.dataset.train.h36m_root, pelvis3d, imageShape=config.image_shape)]
    detector = Detector(config, device)
    prediction, inputBatch = detector.inferHuman36Data(sample, randomize_n_views=config.dataset.val.randomize_n_views,
                                                            min_n_views=config.dataset.val.min_n_views,
                                                            max_n_views=config.dataset.val.max_n_views)
    
    # TODO get the visualization done
    heatmaps_vis = vis.visualize_heatmaps(
                                inputBatch["images_batch"], prediction["heatmaps_pred"],
                                kind=config.kind,
                                batch_index=0, size=5,
                                max_n_rows=10, max_n_cols=10)
    heatmaps_vis = heatmaps_vis.transpose(2, 0, 1)
    for i in range(0,4):
        cv2.imwrite(f"/home/weiwang/Desktop/heatmaps_test_{i}.png", heatmaps_vis[i,:,:])
    