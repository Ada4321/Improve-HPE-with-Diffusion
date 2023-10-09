import argparse
import os
import os.path as path
import zipfile
import numpy as np
import h5py
from glob import glob
from shutil import rmtree
from subprocess import call

import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
from data.h36m_dataset import Human36mDataset
from data.camera import world_to_camera, project_to_2d, image_coordinates
from data.utils import wrap

data_root = '/root/autodl-tmp/human3.6m'
output_filename = 'data_3d_h36m'
output_filename_2d = 'data_2d_h36m_gt'
frames_dir = 'extracted_images/{}/{}/{}'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
cameras = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3
}


if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
    
    if os.path.exists(output_filename + '.npz'):
        print('The dataset already exists at', output_filename + '.npz')
        exit(0)
        
    output = {}
    
    import cdflib
    
    for subject in subjects:
        output[subject] = {}

        cdf_files = glob(data_root + '/' + subject + '/MyPoseFeatures/D3_Positions/*.cdf')
        assert len(cdf_files) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(cdf_files))
        #for f in cdf_files:
        for f in ['/root/autodl-tmp/human3.6m/S1/MyPoseFeatures/D3_Positions/Directions 1.cdf']:
            action = os.path.splitext(os.path.basename(f))[0]
            
            if subject == 'S11' and action == 'Directions':
                continue # Discard corrupted video
                
            # Use consistent naming convention
            canonical_name = action.replace('TakingPhoto', 'Photo') \
                                    .replace('WalkingDog', 'WalkDog')
            
            hf = cdflib.CDF(f)
            positions = hf['Pose'].reshape(-1, 32, 3)
            positions /= 1000 # Meters instead of millimeters
            output[subject][canonical_name] = positions.astype('float32')
    
    print('Saving output_filename...')
    np.savez_compressed(path.join(data_root, output_filename), positions_3d=output)
    
    print('Done.')
        
    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = Human36mDataset(path.join(data_root, output_filename) + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            positions_2d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d
            
    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(path.join(data_root, output_filename_2d), positions_2d=output_2d_poses, metadata=metadata)
    
    print('Done.')

    # Extracting images
    print('')
    print('Extracting images from videos...')

    for subject in dataset.subjects():
        video_files = glob(data_root + '/' + subject + '/Videos/*.mp4')
        for vf in video_files:
            video_name = vf.split('/')[-1]
            action_name = video_name.split('.')[0]
            camera_name = video_name.split('.')[1]

            canonical_action_name = action_name.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')
            canonical_camera_name = cameras[camera_name]

            save_dir = path.join(data_root, frames_dir.format(subject, canonical_action_name, canonical_camera_name))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Use ffmpeg to extract frames into a temporary directory
            call([
                'ffmpeg',
                '-nostats', '-loglevel', 'error',
                '-i', vf,
                '-qscale:v', '3',
                path.join(save_dir, 'img_%06d.jpg')
            ])