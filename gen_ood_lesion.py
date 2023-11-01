### Tumor Generateion
import random
import cv2
import torch
import elasticdeform
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

def to_PIL(tensor):
    tensor = np.stack([tensor, tensor, tensor], axis=-1)
    return Image.fromarray((tensor*255.0).astype(np.uint8))

def to_PIL_raw(tensor):
    return Image.fromarray((tensor*255.0).astype(np.uint8))

def generate_prob_function(mask_shape):
    sigma = np.random.uniform(3,15)
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # Gaussian filter
    # this taks some time
    a_2 = gaussian_filter(a, sigma=sigma)

    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    return a

# first generate 5*200*200*200

def get_texture(mask_shape):
    # get the prob function
    a = generate_prob_function(mask_shape) 

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # if a(x) > random_sample(x), set b(x) = 1
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter

    # Gaussian filter
    if np.random.uniform() < 0.7:
        sigma_b = np.random.uniform(3, 5)
    else:
        sigma_b = np.random.uniform(5, 8)

    # this takes some time
    b2 = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b2 > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b2 * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b2, 0, 1) # 目前是0-1区间
    
    return Bj


# here we want to get predefined texutre:
def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b, 0, 1) # 目前是0-1区间

    return Bj

# Step 1: Random select (numbers) location for tumor.
def random_select(mask_scan):
    # we first find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # we need to strict number z's position (0.3 - 0.7 in the middle of liver)
    z = int(0.5 * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    # erode the mask (we don't want the edge points)
    kernel = np.ones((5,5), dtype=np.uint8)
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

# Step 2 : generate the ellipsoid
def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out

def get_fixed_geo(mask_scan, tumor_type, num_tumor):

    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    # texture_map = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.float16)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32

    if tumor_type == 'tiny':
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'small':
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'medium':
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'large':
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            y = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            z = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            sigma = random.randint(5, 10)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == "mix":
        # tiny
        num_tumor = random.randint(3,10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            y = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            z = random.randint(int(0.75*tiny_radius), int(1.25*tiny_radius))
            sigma = random.uniform(0.5,1)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

        # small
        num_tumor = random.randint(5,10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            y = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            z = random.randint(int(0.75*small_radius), int(1.25*small_radius))
            sigma = random.randint(1, 2)
        
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture
            
        # medium
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            y = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            z = random.randint(int(0.75*medium_radius), int(1.25*medium_radius))
            sigma = random.randint(3, 6)
            
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

        # large
        num_tumor = random.randint(1,3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            y = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            z = random.randint(int(0.75*large_radius), int(1.25*large_radius))
            sigma = random.randint(5, 10)
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x//2, point[1] + enlarge_y//2, point[2] + enlarge_z//2]
            x_low, x_high = new_point[0] - geo.shape[0]//2, new_point[0] + geo.shape[0]//2 
            y_low, y_high = new_point[1] - geo.shape[1]//2, new_point[1] + geo.shape[1]//2 
            z_low, z_high = new_point[2] - geo.shape[2]//2, new_point[2] + geo.shape[2]//2 
            
            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    geo_mask = geo_mask[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    # texture_map = texture_map[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    geo_mask = (geo_mask * mask_scan) >=1
    
    return geo_mask, new_point[2] - enlarge_z//2


def get_tumor(volume_scan, mask_scan, tumor_type, texture, num_tumor):
    geo_mask, z = get_fixed_geo(mask_scan, tumor_type, num_tumor)

    sigma      = np.random.uniform(1, 2)
    difference = np.random.uniform(30, 185) / 255

    # blur the boundary
    geo_blur = gaussian_filter(geo_mask*1.0, sigma)
    abnormally = (volume_scan + texture * geo_blur * difference) * mask_scan
    # abnormally = (volume_scan - texture * geo_mask * difference) * mask_scan
    
    abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask

    return abnormally_full, geo_blur, z

def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture, num_tumor):
    # for speed_generate_tumor, we only send the liver part into the generate program
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # shrink the boundary
    x_start, x_end = max(0, x_start+1), min(mask_scan.shape[0], x_end-1)
    y_start, y_end = max(0, y_start+1), min(mask_scan.shape[1], y_end-1)
    z_start, z_end = max(0, z_start+1), min(mask_scan.shape[2], z_end-1)

    liver_volume = volume_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_mask   = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

    # input texture shape: 420, 300, 320
    # we need to cut it into the shape of liver_mask
    # for examples, the liver_mask.shape = 286, 173, 46; we should change the texture shape
    x_length, y_length, z_length = x_end - x_start, y_end - y_start, z_end - z_start
    start_x = random.randint(0, texture.shape[0] - x_length - 1) # random select the start point, -1 is to avoid boundary check
    start_y = random.randint(0, texture.shape[1] - y_length - 1) 
    start_z = random.randint(0, texture.shape[2] - z_length - 1) 
    cut_texture = texture[start_x:start_x+x_length, start_y:start_y+y_length, start_z:start_z+z_length]


    liver_volume, tumor_mask_blur, z = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture, num_tumor)
    tumor_mask = np.zeros_like(volume_scan)
    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
    tumor_mask[x_start:x_end, y_start:y_end, z_start:z_end] = tumor_mask_blur

    return volume_scan, tumor_mask, z

from tqdm import tqdm


tumor_types = ['tiny', 'small', 'medium', 'large']

textures = []
sigma_as = [3, 6, 9, 12, 15]
# sigma_as = [3]
sigma_bs = [4, 7]
# sigma_bs = [4]
predefined_texture_shape = (420, 300, 320)
all_data = np.memmap('data.mmap', dtype=np.float32, mode='w+', shape=(10, 420, 300, 320))
for sigma_a in sigma_as:
    for sigma_b in sigma_bs:
        texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
        all_data[index] = texture
        print(sigma_a, sigma_b)
        # np.save(f"texture/{sigma_a}_{sigma_b}.npy", texture)

# from tqdm import tqdm
# from copy import deepcopy

# # all_data = np.empty((10, 420, 300, 320), dtype=np.float32)
# index = 0
# for sigma_a in sigma_as:
#     for sigma_b in sigma_bs:
#         # texture = np.load(f"texture/{sigma_a}_{sigma_b}.npy")
#         # for i in tqdm(range(1000)):
#         #     texture = np.load(f"texture/{sigma_a}_{sigma_b}.npz")['data']
#         texture = np.load(f"texture/{sigma_a}_{sigma_b}.npz", mmap_mode='r')['data']
#         # np.savez(f"texture/{sigma_a}_{sigma_b}.npz", data=texture, allow_pickle=True, pickle_kwargs={'protocol': 4})
#         # import ipdb; ipdb.set_trace()

#         all_data[index] = texture
#         print(sigma_a, sigma_b, 3)
#         index += 1 

def run_synthesis(input, tumor_type, num_tumor=1):
    input = input.cpu().squeeze().detach().numpy()
    texture = all_data[random.randint(0, 9)]

    image = np.zeros([256, 256, 64])
    label = np.ones_like(image)
    syn_image, syn_mask, z = SynthesisTumor(image, label, tumor_type, texture, num_tumor)
    syn_image = syn_image[..., z]
    syn_mask = syn_mask[..., z]
    syn_image = np.stack([syn_image, syn_image, syn_image], axis=0)
    syn_mask = np.stack([syn_mask, syn_mask, syn_mask], axis=0)

    # # for CXR
    # ret = input * (1) + syn_image * syn_mask
    # for Fundoscopy (APTOS)
    ret = input * (1) - syn_image * syn_mask
    return torch.from_numpy(ret).cuda().unsqueeze(0).float()
