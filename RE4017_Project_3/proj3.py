"""
Created on Wed Apr 26 08:29:45 2023

RE4017 Project 3

proj3.py

Harris Corner detector and image stitcher

NB Command line argument of img file name including extension required

@author:    Finn Hourigan 17228522
            Ronan Reilly 18242367
            Brendan Lynch 18227651
            Barry Hickey 18243649
"""
import sys
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from math import sqrt

###### FUNCTIONS #############################################################

def rotation_or_scaling_required(check_rotation,check_scale):
    
    #  Ask the user whether to set check_rotation as True or False
    while True:
        user_input = input("Do you want to check for rotation? (Y/N): ")
        if user_input.upper() == "Y":
            check_rotation = True
            break
        elif user_input.upper() == "N":
            check_rotation = False
            break
        else:
            print("Invalid input. Please enter Y or N.")
    
    # Ask the user whether to set check_scale as True or False
    while True:
        user_input = input("Do you want to check for scaling? (Y/N): ")
        if user_input.upper() == "Y":
            check_scale = True
            break
        elif user_input.upper() == "N":
            check_scale = False
            break
        else:
            print("Invalid input. Please enter Y or N.")
    
    # Print the values of check_rotation and check_scale
    print(f"check_rotation = {check_rotation}")
    print(f"check_scale = {check_scale}")
    
    return check_rotation, check_scale

###############################################################################


def harris_response(image, sigma1):

    #convolving with derivative of Gaussian x & y kernels
    I_x = gaussian_filter(image, sigma1, order=(1, 0))
    I_y = gaussian_filter(image, sigma1, order=(0, 1))

    sigma2 = 2.5 * sigma1

    # Convolve with patch filter to find A,B,C as follows
    A = gaussian_filter(I_x * I_x, sigma2 )
    B = gaussian_filter(I_x * I_y, sigma2 )
    C = gaussian_filter(I_y * I_y, sigma2 )
    
    # Find determinant and trace
    det_M = (A * C) - (B * B)
    tra_M = A + C + 0.00001     #addition of 0.00001 avoids possible zero division error

    return det_M / tra_M


###############################################################################


def get_harris_points(harris_im, threshold, min_d):
    #Sample code from PDF applies
    
    # Thresholds the Harris responses and return co-ordinates of HIPs
    
    # Find all HIP candidates above given threshold * max value in 
    corner_threshold = harris_im.max() * threshold
    harris_im_th = (harris_im > corner_threshold)

    # Find the co-ordinates of these candidates and their values
    coords = np.array(harris_im_th.nonzero()).T
    
    # Use co-ordinates to extract these candidate values in the greyscale img
    candidate_values = np.array([harris_im[c[0],c[1]] for c in coords])
    
    # Using argsort, the first element in indices corresponds to the index of 
    # the smallest element in candidate_values
    indices = np.argsort(candidate_values)
   
    # Store allowed point locations in boolean image
    allowed_locations = np.zeros(harris_im.shape, dtype='bool')
    allowed_locations[min_d:-min_d, min_d:-min_d] = True

    # Select the best points, using nonmax suppression based on
    # the array of allowed locations
    filtered_coords = []
    for i in indices[::-1]:
        r,c = coords[i]     
        if allowed_locations[r,c]:
            filtered_coords.append((r,c))
            allowed_locations[r - min_d:r + min_d + 1, c - min_d:c + min_d + 1] = False
            
    
    
    return filtered_coords

###############################################################################

    
def plot_harris_points(image, points):
    # Allows plotting of HIPs superimposed on original image
    
    plt.imshow(image, cmap="gray")
    y, x = np.transpose(points)
    plt.plot(x, y, 'rx')
    plt.axis('off')
    # plt.show()
    plt.close()


###############################################################################

def get_descriptor_vectors(image, HIPs):
    #    1. Form an image patch around the first Harris interest point (11x11 pixels)
    #    2. Flatten it into a 121 element vector.
    #    3. Subtract the mean of that vector from the vector itself
    #    4. Normalize the vector so values range from -1 to +1.
    #    5. Repeat for all interest points, stacking the vectors each time
    
    descriptors = [] #Empty array for descriptor vectors
    
    # If an 11x11 patch is required, set a variable to int 5
    patch_val = 5
    
    for point in HIPs:
        # Slice 11x11 patches at each interest point from the original image
        image_patch = image[point[0] - patch_val : point[0] + patch_val + 1, 
                            point[1] - patch_val: point[1] + patch_val + 1]
        # Flatten the patch values to create a 121 element vector
        desc_vector = image_patch.flatten()
        # Change desc_vector to a zero-mean vector by subtracting the mean
        desc_vector = desc_vector - np.mean(desc_vector) 
        # Normalise each vector then add to descriptors array
        descriptors.append(desc_vector/np.linalg.norm(desc_vector))
    return descriptors

###############################################################################




def get_response_matrix(m1, m2):
    
    #Form the response matrix r12
    
    # m1 refers to a matrix, each row vector corresponding to a harris point in image1
    m1 = np.array(m1)   
    
    # m2 refers to a matrix, each row vector corresponding to a harris point in image2
    m2 = np.array(m2)
    
    #The response matrix r12 is generated as the "outer product" of m1 and m2 transposed
    r12 = np.dot(m1, m2.T)
    
    return r12


###############################################################################


def plot_response_matrix(response_matrix, response_matrix_th):
    
    plt.imshow(response_matrix, cmap='gray')
    plt.show()
    
    
    plt.imshow(response_matrix_th, cmap='gray')
    plt.show()
   
###############################################################################


def find_matches (r12,threshold): 
    r12_th = r12.copy()
    
    # Initialise array for co-ordinates of Harris points that match between the 2 images
    matches = []
    # Loop through all elements of r12, storing any elements above the given threshold in 
    # the ' matches' array
    for r in range(r12.shape[0]):
        for c in range(r12.shape[1]):
            if r12[r][c] > threshold:
                 matches.append((r,c))
            #thresholded array formed, setting values below threshold to zero    
            else: r12_th[r][c] = 0  
    
    #plot_response_matrix(r12, r12_th)
   
    # NB: note that these are just specifying which descriptor vectors match each other
    # but don't specify the co-ordinates
    
    return  matches


###############################################################################


def plot_matches(matched_coordinate_pairs, im_1, im_2, HIPs_im_1, HIPs_im_2):
    
    # Plot the mapping of matching co-ordinates, indicating the general consensus
    # for the correct translation
    
    ## Check if image dimensions are the same..
    
    if im_1.shape != im_2.shape:
        print("im_1.shape != im_2.shape")
        # if image 1 is smaller, pad image 1 with zeros
        if im_1.shape < im_2.shape:
            # Generate a zeros array same dimensions as the larger image
            padded_im_1 = np.zeros(im_2.shape)
            # Then "paste" the array of the smaller image into the zeros array
            padded_im_1[:im_1.shape[0], :im_1.shape[1]] = im_1
            # Combine the two arrays, now of the same dimensions
            combined_im = np.concatenate((padded_im_1, im_2), axis=1)
        # otherwise, pad image 2 with zeros
        else:
            padded_im_2 = np.zeros(im_1.shape)
            padded_im_2[:im_2.shape[0], :im_2.shape[1]] = im_2
            combined_im = np.concatenate((im_1, padded_im_2), axis=1)
            
    #If they do share the same dimensions, simply concatenate them side by side
    else: combined_im = np.concatenate((im_1, im_2), axis=1)
    
    plt.imshow(combined_im, cmap='gray')
    
    # Find number of matched pairs between images
    num_matched_pairs = np.array(matched_coordinate_pairs).shape[0]
    
    # plot matched pairs
    for i in range(num_matched_pairs):
        
        c1, c2 = matched_coordinate_pairs[i]
        
        plt.plot([HIPs_im_1[c1][1], HIPs_im_2[c2][1] + im_1.shape[0]],
                 [HIPs_im_1[c1][0], HIPs_im_2[c2][0]], 'y', linewidth=1)
        
        plt.plot(HIPs_im_1[c1][1], HIPs_im_1[c1][0], 'ro', markersize=3)
        plt.plot(HIPs_im_2[c2][1] + im_1.shape[0],
                 HIPs_im_2[c2][0], 'bo', markersize=3)

    
    
    plt.show()
    plt.close()

###############################################################################


def get_best_translation_RANSAC (corresponding_HIP_pairs, HIPs_image_1, HIPs_image_2, RANSAC_threshold):
    # USe RANSAC alg to find strongest mapping between image 1 and image 2.
    
    #Initiate np array for storing translations, storing pairs of r,c co-ordinates
    translations = np.zeros((len(corresponding_HIP_pairs), 2))
    
    for i in range(len(corresponding_HIP_pairs)):
                
        p1,p2 = corresponding_HIP_pairs[i]
        # p1 identifies an interest point location in the array HIPs_image_1 that 
        # corresponds with the interest point identified by p2 in HIPs_image_2
        
        # In other words, the calculations thus far suggest that these two 
        # points should 'overlap' when the final combined image is generated 
        
        # find translation of corresponding points:
        
        # find row translations for translation i
        translations[i][0] = HIPs_image_1[p1][0] - HIPs_image_2[p2][0]
        #find column translation for translation i
        translations[i][1] = HIPs_image_1[p1][1] - HIPs_image_2[p2][1]
    
    # Initialise array for storing how many agreements each translation has
    ransac_agreements = []
    
    # For each translation in the list....
    for i in range(len(corresponding_HIP_pairs)):
        agreements = 1 # 1 being itself
        
        r_translation = translations[i][0]
        c_translation = translations[i][1]

        
        # Compare each translation, with every other translation
        for k in range(len(corresponding_HIP_pairs)):
            # Not including itself
            if k!= i:
                r_translation_compare = translations[k][0]
                c_translation_compare = translations[k][1]
                # Calculate error
                euclidean_distance = sqrt((r_translation - r_translation_compare) ** 2
                                          + (c_translation - c_translation_compare) ** 2)
                # If the error is less than the threshold, increment agreements
                if euclidean_distance <= RANSAC_threshold:
                    
                    agreements += 1
        
        # Store number of agreements for translation i in ransac_agreements array                         
        ransac_agreements.append(agreements)    
    
    # Index of translations with most agreements identified
    best_translation_index = np.argmax(ransac_agreements)
    best_translation = translations[best_translation_index]
    most_agreements = ransac_agreements[best_translation_index]
    
    print(f"Number of strong matches:   {most_agreements}")
    
    dr = best_translation[0]
    dc = best_translation[1]

    return dr, dc, most_agreements


###############################################################################




def scale_array(img_array,scale):
    # Scale array by given scale
    height, width = img_array.shape
    
    # Calculate the new size after scaling 
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Resize the image using Pillow
    image = Image.fromarray((img_array*255).astype(np.uint8))
    image = image.resize((new_width, new_height))
    
    # Convert the resized image back to a numpy array
    img_array = np.array(image)/255.0
    
    return img_array





###############################################################################



def scale_image(image,scale):
    # Scale image by given scale
    
    # Convert to array
    img_array = np.array(image)/255
        
    # RGBA image array requires 3 variables on LHS
    height, width, channels = img_array.shape
    
    # Calculate the new size after scaling  
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Resize the image using Pillow
    image = image.resize((new_width, new_height))
          
    return image
###############################################################################


def compose_images(image1, image2, dr, dc):
    # Composition of final images on a white background
   
    # Case: 1
    if dr > 0 and dc > 0:
        canvas = Image.new('RGB', (int(dc) + image2.width, int(dr) + image2.height), color='white')
        canvas.paste(image2, (int(dc), int(dr)))
        canvas.paste(image1, (0,0))
    
    # Case: 2
    elif dr < 0 and dc > 0:
        canvas= Image.new('RGB', (int(dc) + image2.width, int(abs(dr)) + image1.height), color='white')         
        canvas.paste(image2, (int(dc), 0))
        canvas.paste(image1, (0, int(abs(dr))))
    
    # Case: 3
    elif dr > 0 and dc < 0:   
        canvas = Image.new('RGB', (int(abs(dc)) + image1.width, int(dr) + image2.height), color='white')
        canvas.paste(image2, (0, int(dr)))
        canvas.paste(image1, (int(abs(dc)), 0))
    
    # Case: 4
    else:  
        canvas = Image.new('RGB', (int(abs(dc)) + image1.width, int(abs(dr)) + image1.height), color='white')
        canvas.paste(image2, (0,0))
        canvas.paste(image1, (int(abs(dc)), int(abs(dr))))
        
    plt.imshow(canvas)
    plt.axis('off')
    plt.show()




###############################################################################

def find_translation(scale,angle):
    
    ### STEP 2.1 ###########################################
    
    # Open images, convert to grayscale and normalizeby dividing by 255
    image_1 = np.array(Image.open(f"./Test Images/{img1_name}").convert('L'))/255.0
    image_2 = np.array(Image.open(f"./Test Images/{img2_name}").convert('L'))/255.0
    
    
    
    # Rotate image 2 by given angle
    image_2 = ndimage.rotate(image_2, angle, reshape=False, order=0)
    
    # Scale image 2 by given scale
    image_2 = scale_array(image_2,scale)
    
    print("")
    print(f"Angle = {angle} degrees")
    print(f"Scale = {scale} ")
            
    ##### Step 2.2. Form Harris responses & find Harris interest points (HIPs)
    harris_response_image_1 = harris_response(image_1, sigma1 = 1)
    harris_response_image_2 = harris_response(image_2, sigma1 = 1)
    
    HIPs_image_1 = get_harris_points(harris_response_image_1, threshold=HIP_th, min_d=10)
    HIPs_image_2 = get_harris_points(harris_response_image_2, threshold=HIP_th, min_d=10)
    
    print(f"Number of Harris points (img1):   {len(HIPs_image_1)}")
    print(f"Number of Harris points (img2):   {len(HIPs_image_2)}")
    
                
    # Step 2.3 Form normalized 121 element patch descriptor vectors
    descriptors_image_1 = get_descriptor_vectors(image_1, HIPs_image_1)
    descriptors_image_2 = get_descriptor_vectors(image_2, HIPs_image_2)

    # Step 2.4. Generate response matrix from desciptor vectors
    r12 = get_response_matrix(descriptors_image_1, descriptors_image_2)
    matched_HIP_pairs = find_matches (r12 , threshold = matching_th)
    
    if len(matched_HIP_pairs) > 0:
        
        # Step2.5 . Use RANSAC algorithm for finding best translation and number of strong agreements
        dr, dc, most_agreements = get_best_translation_RANSAC(matched_HIP_pairs, HIPs_image_1,
                                          HIPs_image_2, RANSAC_threshold=1.6)
    else:
        # Dummy values assigned if no matched HIPs found
        dr, dc, most_agreements = 0,0,0
    
        
    return dr,dc, most_agreements




###############################################################################

#######    MAIN    ############################################################


if __name__ == "__main__":
    
    ### STEP 1 Ask user if rotation or scaling checks required #####
    
    check_rotation = True # default values for check_rotation and check_scale
    check_scale = True
    
    check_rotation,check_scale = rotation_or_scaling_required(check_rotation,check_scale)
    
             
    if (check_scale):
        scales = np.linspace(0.5, 1.5, num=11, endpoint=True)

    else:
        scales = [1]

        
    if (check_rotation):
        angles = list(range(-45, 50, 5))
    else:
        angles = [0]
   
    # Harris Interest Point threshold value
    HIP_th = 0.4
    
    # Matching descriptor vectors threshold
    matching_th = 0.95
    
    img1_name = sys.argv[1]
    img2_name = sys.argv[2]
    
    # img1_name = "tigermoth1.png"########
    # img2_name = "tigermoth2.png"
    
    # img1_name = "arch1.png"######
    # img2_name = "arch2.png"
    
    # img1_name = "arch1.png"######
    # img2_name = "arch2_rotated.png"
    
    # img1_name = "arch1.png"#######
    # img2_name = "arch2_scaled.png"
    
    # img1_name = "arch1.png"#####
    # img2_name = "arch2_rotated_and_scaled.png"
    
    # img1_name = "balloon1.png"#####
    # img2_name = "balloon2_rotated.png"
    
    # img1_name = "balloon1.png"#####
    # img2_name = "balloon2.png"
    
    # img1_name = "balloon1.png"#######
    # img2_name = "balloon2_scaled.png"
    
    # img1_name = "balloon1.png"####
    # img2_name = "balloon2_rotated_and_scaled.png"  
    
    # Arrays for storing the matches, angle used and scale used for each iteration
    agreements_at_given_angle_and_scale = []
    angles_list = []
    scales_list = []
   
    for scale in scales:
        
        for angle in angles:
            
            ## Step 2 Find strong agreements at given angle and scale
            dr,dc,most_agreements = find_translation(scale,angle)
            
            ## Step 3. Storing of iteration results
            agreements_at_given_angle_and_scale.append(most_agreements)
            angles_list.append(angle)
            scales_list.append(scale) 
    
    # boolean initialised to False
    valid_match = False
    
    # Step 4. Determining iteration which produced the best match
    
    # check that at least one iteration resulted in at least one strong match
    for num in agreements_at_given_angle_and_scale:
    
        if num > 0:
            valid_match = True
            break     
    
    if (valid_match):
        
        best_match_index = np.argmax(agreements_at_given_angle_and_scale)
        
        best_angle = angles_list[best_match_index]
        best_scale = scales_list[best_match_index]
        
        print("")
        print(f"Best rotation = {best_angle} degrees")
        print("")
        print(f"Best scale = {best_scale} ")
        print("")
        
        # Step 5. Find best translation 
        dr, dc, most_agreements = find_translation(best_scale,best_angle)
        
        # Step 6. Transform image objects as required
        final_image_1 = Image.open(f"./Test Images/{img1_name}")
        
        final_image_2 = Image.open(f"./Test Images/{img2_name}")
        final_image_2 = final_image_2.rotate(best_angle)
        final_image_2 = scale_image(final_image_2, best_scale) 
        
        #Step 7 Compose images
        compose_images(final_image_1, final_image_2, dr, dc)
        
        
    else:
        print("No matches found!")
        

    
