import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import trimesh
import json


# Crops an image
#
# OpenCV coordinate system
#  |--------> u
#  |
# \|/ v
#
# Cropping coordinate system for an image as numpy array (x,y,3):
#  |--------> y
#  |
# \|/ x
#

def crop(origin_u:int, origin_v:int, width:int, height:int, img):
    # Sanity checks
    if origin_u < 0:
        raise Exception("Invalid origin u")
    if origin_v < 0:
        raise Exception("Invalid origin v")
    if (origin_u + width) > img.shape[1]:
        raise Exception("Invalid cropping width")
    if (origin_v + height) > img.shape[0]:
        raise Exception("Invalid cropping height")
    
    return img[origin_v:(origin_v + height), origin_u:(origin_u + width)]

def append_point(meshes_list, radius, translation, color):
    point_mesh = trimesh.creation.icosphere(radius=radius)
    if translation is not None:
        point_mesh.vertices = point_mesh.vertices + translation
    point_mesh.visual.face_colors = color
    meshes_list.append(point_mesh)

def correct_meshroom_extrinsics(camera_rotation_meshroom, camera_center_meshroom):
    rotation_180_around_x_correction = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float32)
    center = np.matmul(rotation_180_around_x_correction, camera_center_meshroom)
    rotation = np.matmul(camera_rotation_meshroom, rotation_180_around_x_correction)
    return rotation, center

def is_fully_visible(object_points, center, target_mesh, tol=1.0e-8):
    # Calculate rays to the object points
    ray_directions = object_points - center.transpose()
    # Project on the bridge
    points_count = object_points.shape[0]
    ray_origins = np.tile(center, (points_count,)).transpose()
    # TODO use index_tri to guess what is in the middle
    locations, index_ray, _ = target_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions)
    # Pick the closest point of each ray
    location_distances = np.linalg.norm(locations - center.transpose(), axis=1).reshape(-1, 1)
    closest_locations = np.zeros((points_count, 3))
    for i in range(points_count):
        closest_locations[i, :] = locations[index_ray == i][np.argmin(location_distances[index_ray == i])]
    # If the closest locations are == to the initial object_points, then it is visible
    diff = closest_locations - object_points
    dist = np.linalg.norm(diff.ravel())
    return dist < tol


def slice_mesh_with_fuse(rotation, center, camera_matrix, image_width, image_height, mesh):
    """
    @brief Slice a mesh according to the fuse of a camera
    @param rotation Camera rotation matrix
    @param center   Camera center in world coordinates
    @param mesh     Target mesh to slice
    @return         A new mesh with only the portion inside the camera fuse
    """

    v = mesh.vertices.copy()
    f = mesh.faces.copy()

    # Extreme pixels in homogenous coordinates
    extremes = np.float32([
        [0.0, image_width, image_width, 0.0],
        [0.0, 0.0, image_height, image_height],
        [1.0, 1.0, 1.0, 1.0]
    ])

    camera_matrix_inv = np.linalg.inv(camera_matrix)
    extremes_3d = np.matmul(camera_matrix_inv, extremes)

    for j in range(extremes_3d.shape[1]):
        point0 = extremes_3d[:, j - 1]
        point1 = extremes_3d[:, j]
        point2 = np.float32([0, 0, 0])
        normal_to_camera = np.cross(point2 - point1, point0 - point1)
        normal_to_world = np.matmul(rotation.transpose(), normal_to_camera).flatten()
        v, f, _ = trimesh.intersections.slice_faces_plane(v, f, normal_to_world, center.flatten())
        
    return trimesh.Trimesh(vertices=v, faces=f)



def masks_aggregate(masks:list[np.ndarray], is_positive:list[bool]):
    """
    @brief Aggregate N masks according to their type (positive or negative) and their order
    @detail Type: "positive" type means "to be kept"; "negative" means "to not be kept"
            The order along the depth is ascending, i.e. 0 for the most close to the observer.
            For example, masks[i] is the mask of an object closer than the object of masks[j], where i < j
    @param masks A set of N grayscale images as numpy array, the depth is the length of the set
    @param masks_types An array (1 x N), True is "positive" and False is "negative"
    @return The aggregation of the masks
    """
    
    if len(masks) != len(is_positive):
        raise Exception("Inconsistent masks and positiveness list lengths (%d, %d)", \
                        (len(masks), len(is_positive)))
    
    if len(masks) == 0:
        raise Exception("No masks")

    width = (masks[0]).shape[1]
    height = (masks[0]).shape[0]

    # Make a copy of the original masks, so they remain untouched
    working_masks = masks.copy()
    for k in range(len(working_masks)):

        # Sanity check on dimesions
        if (masks[k]).shape[1] != width:
            raise Exception("Inconsistent mask width (%d)" % k)
        if (masks[k]).shape[0] != height:
            raise Exception("Inconsistent mask height (%d)" % k)
        
        # if it is a negative masks,
        # then it must be deleted on all the following positive masks (see kk index)
        if is_positive[k] == False:
            _, neg_mask_k = cv.threshold(working_masks[k], 128, 255, cv.THRESH_BINARY_INV)
            for kk in range(k + 1, len(working_masks)):
                if is_positive[kk] == True:
                    # Apply negative masks[k] to positive masks[kk].
                    # Logical table for applying a "negative" mask k to a "positive" mask kk with kk > k:
                    # mask[k] | mask[kk] | result
                    # ---------------------------
                    #   0     |    0     |  0
                    #   0     |    1     |  1
                    #   1     |    0     |  0
                    #   1     |    1     |  0
                    # So the result is (-masks[k]) AND (-masks[kk]).
                    # In other words, the white masks[k] pixels must create a black hole in the masks[kk]
                    working_masks[kk] = cv.bitwise_and(neg_mask_k, working_masks[kk])

    # Once all the negative masks are applied to the positive masks,
    # then the positive masks are aggregated with a logical OR
    result = np.zeros((height, width), dtype=np.uint8)
    for k in range(len(working_masks)):
        if is_positive[k] == True:
            result = cv.bitwise_or(result, working_masks[k])

    return result


def sort_meshes_along_camera_z(rotation, translation, meshes):
    # To enforce the correct dimension
    tvec = translation.reshape(3, 1)
    centroid_distances = np.zeros((1, len(meshes)), dtype=np.float32).flatten()
    for i in range(len(meshes)):
        mesh = meshes[i]
        centroid_world = mesh.centroid
        centroid_camera = np.matmul(rotation, centroid_world.reshape(3, 1)) + tvec
        centroid_distance_z = np.abs(centroid_camera[2, 0])
        centroid_distances[i] = centroid_distance_z
    
    print(centroid_distances)
    return np.argsort(centroid_distances)



def draw_mask_on_img(mask, img, alpha = 0.6, color = (255, 0, 0)):
    """
    @brief Draws a mask on a 3-channel image using a given alpha and a give color
    """
    _, mask_inv = cv.threshold(mask, 128, 255, cv.THRESH_BINARY_INV)
    background = cv.bitwise_and(img, img, mask=mask_inv)

    # Blend image and mask
    film = np.zeros(img.shape, dtype=np.uint8)
    film[:] = color
    beta = 1.0 - alpha
    blended = cv.addWeighted(film, alpha, img, beta, gamma=0)
    foreground = cv.bitwise_and(blended, blended, mask=mask)
    img_with_mask = cv.add(foreground, background)
    return img_with_mask

def create_masks_from_meshes(meshes:list[np.ndarray], img:np.ndarray, rotation:np.ndarray, center:np.ndarray, camera_matrix:np.ndarray, dist_coeffs:np.ndarray) -> list[np.ndarray]:
    """
    @brief Projects N meshes onto an image creating a list of N masks
    """
    
    img_height = img.shape[0]
    img_width = img.shape[1]

    masks = []
    # project all vertices
    for mesh in meshes:
        tvec = -np.matmul(rotation, center)
        rvec, _ = cv.Rodrigues(rotation)
        # If experienced some problems with distorsion coefficient and points very much out of the image,
        # then keep a zero distorsion with:
        # zero_distorsion = np.array([0, 0, 0, 0], dtype = np.float32)
        # This problem should be solved if the meshes are cut with a camera fuse
        pixels_projected, _ = cv.projectPoints(mesh.vertices, rvec, tvec, camera_matrix, dist_coeffs)
        pixels_projected = pixels_projected.squeeze().astype(np.int32)
        # build an ideal grayscale image
        u_min = min(0, np.min(pixels_projected[:, 0]))
        u_max = max(img_width - 1, np.max(pixels_projected[:, 0]))
        v_min = min(0, np.min(pixels_projected[:, 1]))
        v_max = max(img_height - 1, np.max(pixels_projected[:, 1]))
        # Big mask containing all in grayscale
        big_mask_width = u_max - u_min + 1
        big_mask_height = v_max - v_min + 1
        print(f"Creating mask with {big_mask_width, big_mask_height} shape")
        big_mask = np.zeros((big_mask_height, big_mask_width), dtype=np.uint8)
        # Translation vector moving the origin: how the big mask origin sees the original origin
        t_big_mask = np.int32([-u_min, -v_min])
        # Translate to have all the projected pixels positive in (u, v)
        pixels_projected_t = pixels_projected + t_big_mask
        # Draw the triangles
        for t in range(mesh.faces.shape[0]):
            poly = pixels_projected_t[mesh.faces[t, :]]
            cv.drawContours(image=big_mask, contours=[poly], contourIdx=0, color=255, thickness=-1)

        mask = crop(t_big_mask[0], t_big_mask[1], img_width, img_height, big_mask)
        masks.append(mask)

    return masks


def get_image_data_from_json(image_name, json_file_path):
    """ Loads image extrinsics using image name and given json file with sertain structure"""
    #load json as dict 
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    poses = data['poses']
    views = data['views']
    viewId = None
    poseId = None
    frameId = None
    intrinsicId = None
    
    for v in views:
        if v['path'].split('/')[-1] == image_name:
            viewId = v['viewId']
            poseId = v['poseId']
            frameId = v['frameId']
            intrinsicId = v['intrinsicId']
            # print(v['path'])
    if viewId is not None:
        for p in poses:
            if p['poseId'] == poseId:
                rotation = p['pose']['transform']['rotation']
                center = p['pose']['transform']['center']
    else:
        raise Exception(f'Image {image_name} is not found in {json_file_path}')
    
    return viewId, poseId, frameId, intrinsicId, rotation, center


def get_camera_intrinsics_from_json(image_name, json_file_path):
    """ Loads image extrinsics using image name and given json file with sertain structure"""
    viewId, poseId, frameId, intrinsicId, _, _ = get_image_data_from_json(image_name, json_file_path)
    #load json as dict 
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    intrinsics = data['intrinsics']
    for i in intrinsics:
        if i['intrinsicId'] == intrinsicId:
            width = float(i['width'])
            height = float(i['height'])

            sensor_width = float(i['sensorWidth'])
            sensor_height = float(i['sensorHeight'])
            focal_length = float(i['focalLength'])

            offset_x = float(i['principalPoint'][0])
            offset_y = float(i['principalPoint'][1])
            fx = width / sensor_width * focal_length

            fy = height / sensor_height * focal_length
            cx = offset_x + width * 0.5
            cy = offset_y + height * 0.5
            k1 = float(i['distortionParams'][0])
            k2 = float(i['distortionParams'][1])
            k3 = float(i['distortionParams'][2])
            dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype = np.float32)

            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]
            ], dtype = np.float32)
            return camera_matrix, dist_coeffs