# %%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import trimesh

# %%
# plt.rcParams['figure.figsize'] = [25, 15]
plt.rcParams['figure.figsize'] = [12.5, 7.5]

# %%
# ===============================================================
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

# %%
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

# Test for masks_aggregate
tree = np.array([
    [0, 0, 255, 0],
    [0, 0, 255, 0],
    [0, 0, 255, 0],
    [0, 0, 255, 0]
], dtype=np.uint8)

pier0 = np.array([
    [0, 255, 255, 255],
    [0, 255, 255, 0],
    [0, 255, 255, 0],
    [0, 255, 255, 0]
], dtype=np.uint8)

pier1 = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 255, 255],
    [0, 0, 255, 255]
], dtype=np.uint8)

test_masks = [tree, pier0, pier1]
test_masks_types = [False, True, True]
test_masks_aggregated = masks_aggregate(test_masks, test_masks_types)
print(test_masks_aggregated)
# Supposed value:
# [[  0 255   0 255]
#  [  0 255   0   0]
#  [  0 255   0 255]
#  [  0 255   0 255]]

# %%
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

# Test sort_meshes_along_camera_z
cube0 = trimesh.creation.box([1.0, 1.0, 1.0], np.diag(np.full(4, 1)))

translate_cube_vec = np.array([2.0, 0.0, 0.0], dtype=np.float32)

cube1 = cube0.copy()
cube1.vertices += translate_cube_vec

cube2 = cube0.copy()
cube2.vertices -= translate_cube_vec

# y
# |
# --------------------------------> x
#  -2          0          2
# cube2       cube0      cube1

#  0  -1  0
#  0   0  -1
#  1   0  0
rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
translation = np.array([0, 0, 4], dtype=np.float32).reshape(3, 1)

sort_arg = sort_meshes_along_camera_z(rotation, translation, [cube0, cube1, cube2])
# expected 1, 0, 2
print(sort_arg)



# %%
# Intrinsics (from Meshroom)
width = 5472
height = 3648

sensor_width = 13.199999809265137
sensor_height = 8.8000001907348633
focal_length = 10.650865779474191

offset_x = 16.406135714287871
offset_y = -35.738027598766635

fx = width / sensor_width * focal_length
fy = height / sensor_height * focal_length
cx = offset_x + width * 0.5
cy = offset_y + height * 0.5
k1 = 0.014238337366878738
k2 = 0.025684230040371497
k3 = -0.035024634278687523
dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype = np.float32)

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype = np.float32)

# %%
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


def create_masks_from_meshes(meshes:list[np.ndarray], img:np.ndarray, rotation:np.ndarray, center:np.ndarray) -> list[np.ndarray]:
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

# %%
# Extrinsics (from Meshroom)
rotation_0019 = np.array([
    0.91313380665093469,
    0.0021401490144515138,
    0.40765435225613539,
    0.014253132426165696,
    0.99920721813162827,
    -0.037172347917435317,
    -0.40741072564086428,
    0.039753679022810157,
    0.91237938689830989
], dtype=np.float32).reshape(3, 3).transpose()

center_0019 = np.array([
    -0.87003242195187902,
    3.0850863601623053,
    0.8239427362297933
], dtype=np.float32).reshape(3, 1)

rotation_0019, center_0019 = correct_meshroom_extrinsics(rotation_0019, center_0019)

# Projection object
pier = trimesh.load("data/pier.obj", force='mesh')
print("pier.vertices.shape")
print(pier.vertices.shape)
# original image used for the photogrametry
img_0019 = cv.imread("data/DJI_0019_original.jpg")
img_0019 = cv.cvtColor(img_0019, cv.COLOR_BGR2RGB)
print("img_0019.shape is (%d, %d)" % (img_0019.shape[0], img_0019.shape[1]))
masks_0019 = create_masks_from_meshes([pier], img_0019, rotation_0019, center_0019)
mask_0019 = masks_0019[0]
print(mask_0019.shape)

# create figure 
fig = plt.figure(figsize=(25, 15)) 

# Adds a subplot at the 1st position 
rows = 1
columns = 2
fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(mask_0019, 'gray') 
plt.axis('off') 
plt.title("Mask")
  
# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
# showing image 
plt.imshow(draw_mask_on_img(mask_0019, img_0019)) 
plt.axis('off') 
plt.title("Image with mask") 

plt.show()

# %%
# Extrinsics (from Meshroom)
rotation_0028 = np.array([
    0.91065375796888892,
    0.0040475181441892976,
    0.41315051820615323,
    0.020225371148725295,
    0.99831652529554538,
    -0.054360368686439328,
    -0.41267501433820392,
    0.057859596599887916,
    0.90903883284613407
], dtype=np.float32).reshape(3, 3).transpose()

center_0028 = np.array([
    -0.8164042855596807,
    0.55535017750981086,
    0.60915550417724662
], dtype=np.float32).reshape(3, 1)

rotation_0028, center_0028 = correct_meshroom_extrinsics(rotation_0028, center_0028)

# Projection object
pier = trimesh.load("data/pier.obj", force='mesh')
pier_cap = trimesh.load("data/pier_cap.obj", force='mesh')
main_longitudinal_girders = trimesh.load("data/main_longitudinal_girders.obj", force='mesh')
# original image used for the photogrametry
img_0028 = cv.imread("data/DJI_0028_original.jpg")
img_0028 = cv.cvtColor(img_0028, cv.COLOR_BGR2RGB)
main_longitudinal_girders = slice_mesh_with_fuse(rotation_0028, center_0028, camera_matrix, img_0028.shape[1], img_0028.shape[0], main_longitudinal_girders)
pier_sliced = slice_mesh_with_fuse(rotation_0028, center_0028, camera_matrix, img_0028.shape[1], img_0028.shape[0], pier)
pier_cap_sliced = slice_mesh_with_fuse(rotation_0028, center_0028, camera_matrix, img_0028.shape[1], img_0028.shape[0], pier_cap)
masks_0028 = create_masks_from_meshes([main_longitudinal_girders, pier_sliced, pier_cap_sliced], img_0028, rotation_0028, center_0028)
mask_0028 = masks_aggregate(masks_0028, [True, True, True])
print(mask_0028.shape)
cv.imwrite('data/mask_0028.png', mask_0028)
image_written = cv.imread('data/mask_0028.png', cv.IMREAD_UNCHANGED)
print(image_written.shape)
if len(image_written.shape) == 2:
    print("is grayscale")
else:
    print("Image is NOT grayscale, len = %d" % len(image_written))

# create figure 
fig = plt.figure(figsize=(25, 15)) 

# Adds a subplot at the 1st position 
rows = 1
columns = 2
fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(mask_0028, 'gray') 
plt.axis('off') 
plt.title("Mask")

# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
# showing image 
plt.imshow(draw_mask_on_img(mask_0028, img_0028)) 
plt.axis('off') 
plt.title("Image with mask") 

plt.show()

# %%
# Visualise the scene ------------------------------------------------------------------
meshes = []
# # Append object points
# for i in range(object_points.shape[0]):
#     append_point(meshes, 0.05, object_points[i, :], (0, 0, 255))

# World rig
world_xyz = trimesh.creation.axis()
meshes.append(world_xyz)
# Camera rig 0019
camera_xyz_0019 = trimesh.creation.axis()
camera_xyz_0019.vertices = (np.matmul(rotation_0019.transpose(), camera_xyz_0019.vertices.transpose()) + center_0019).transpose()
meshes.append(camera_xyz_0019)
# Camera rig 0028
camera_xyz_0028 = trimesh.creation.axis()
camera_xyz_0028.vertices = (np.matmul(rotation_0028.transpose(), camera_xyz_0028.vertices.transpose()) + center_0028).transpose()
meshes.append(camera_xyz_0028)

# Pier mesh
meshes.append(pier)
meshes.append(pier_cap)

# Create scene
scene = trimesh.Scene(meshes)
scene.show('notebook')

# %%
# Visualise the scene with slices ------------------------------------------------------------------
meshes = []

# World rig
world_xyz = trimesh.creation.axis()
meshes.append(world_xyz)
# Camera rig 0019
camera_xyz_0019 = trimesh.creation.axis()
camera_xyz_0019.vertices = (np.matmul(rotation_0019.transpose(), camera_xyz_0019.vertices.transpose()) + center_0019).transpose()
meshes.append(camera_xyz_0019)
# Camera rig 0028
camera_xyz_0028 = trimesh.creation.axis()
camera_xyz_0028.vertices = (np.matmul(rotation_0028.transpose(), camera_xyz_0028.vertices.transpose()) + center_0028).transpose()
meshes.append(camera_xyz_0028)

# Pier mesh
pier_cutted = slice_mesh_with_fuse(rotation_0019, center_0019, camera_matrix, 5472, 3648, pier)
meshes.append(pier_cutted)
meshes.append(pier_cap)

# Create scene
scene = trimesh.Scene(meshes)
scene.show('notebook')



