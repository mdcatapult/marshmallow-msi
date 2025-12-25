import sys
import gc
import numpy as np
import pathlib
import gc
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('/Users/ana-maria.nastase/git/MDCP-0461-Holmium/code')
from process_imat import Marshmallow
import imat
import utils


def get_masks(mask_folder, string_to_find='Trans*_wt*'):
    dict_mask = {}
    entries = list(pathlib.Path(mask_folder).glob('*'))
    for entry in entries:
    
        file = list(pathlib.Path(entry).glob(string_to_find))
        n_entry = len(str(entry).split('/'))-1
        id = str(entry).split('/')[n_entry]
        # print(id, file)
        if len(file)>0:
            mask = plt.imread(file[0])
            # plt.imshow(mask)
            # plt.show()
            dict_mask[int(id)] = mask
    dict_mask = dict(sorted(dict_mask.items()))

    return dict_mask


def extract_pixels(mask, msn):
   
    tbi_labels = np.where(mask == 255)
    no_tbi_labels = np.where(mask == 0)

    new_imat = msn.int_mat  
    new_imat[no_tbi_labels] = 0
    new_msn = imat.imat(new_imat, msn.mz)
    new_msn.set_labels(no_tbi_labels, tbi_labels)
    return new_msn

def get_mask_contour(mask, dilate=False):
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        mask = cv2.dilate(mask, kernel, iterations=2)
    _, threshold = cv2.threshold(mask, 0.1, 255, cv2.THRESH_BINARY) 
    # Detecting contours in image. 
    contours, _= cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    xs,ys = [],[]
    for i in contours:
        for k in i:
            x,y = k[0]
            xs.append(x)
            ys.append(y)
    return xs,ys

def sample_random_pxl(wt_labels, n_tbi_pixels=10):
    rand_sample = random.sample(range(0,len(wt_labels[0])),n_tbi_pixels)
    random_wt_labels = (np.array([wt_labels[0][r] for r in rand_sample]),
                        np.array([wt_labels[1][r] for r in rand_sample]))
    return random_wt_labels

def get_wt_labels(tbi_mask, wt_labels, tbi_labels):
    tbi_wt_mask = np.zeros((tbi_mask.shape))
    tbi_wt_mask[wt_labels] = 1
    tbi_wt_mask[tbi_labels] = 2
    final_labels = np.where(tbi_wt_mask == 1)
    return final_labels

def gauss(dist,sigma):
    """Calculate the distribution used for calculating the weight of the particles."""

    gauss_prob = (-0.5)*(dist**2)/sigma**2 #/(sigma*np.sqrt(2*math.pi))

    return gauss_prob

def norm_weights(weight_list):
      #normalize weights vector to sum 1
    weight_list = np.array(weight_list)
    weight_list = np.exp(weight_list-weight_list.max())
    weight_list /= weight_list.sum()
    return weight_list

def plot_roi_ions(msn, median, xs,ys):
     
    # p = list(np.abs(median))
    # indices, p_sorted = zip(*sorted(enumerate(p), key=itemgetter(1))[::-1])
    # #  print(len(p_sorted[p_sorted > 0]))
    # p_sorted = p_sorted/np.max(p_sorted)
    # print(len(np.nonzero(p_sorted)[0]))
    # print('p_sorted',p_sorted[:5])
    p_sorted = median[4]
    indices = median[3]
    dif = median[5]
    filt = median[2]
    # plt.hist([x[0] for x in filt])
    # plt.show()
    wilcox = stats.wilcoxon([x[0] for x in filt], [x[1] for x in filt])
    wilcox_con = stats.wilcoxon([x[2] for x in filt], [x[1] for x in filt])
    logfc = np.log2(np.divide(np.mean([x[0]+1 for x in filt]), np.mean([x[1]+1 for x in filt])))
    logfc_con = np.log2(np.divide(np.mean([x[2]+1 for x in filt]), np.mean([x[1]+1 for x in filt])))
    print(wilcox)
    adj_pval = mult_test(wilcox.pvalue, alpha=0.05, method='fdr_tsbky')[1]
    wilcox_res.append((wilcox.pvalue, wilcox_con.pvalue, logfc, logfc_con,len(dif), np.std([x[0] for x in filt]), np.std([x[1] for x in filt])))
        
    if wilcox.pvalue < 0.05:
        print('significant', adj_pval)
    

    # for i,ind in enumerate(indices[:5]):
        
    #     print(ind, msn.mz[ind])
    #     ion = msn.int_mat[:,:,ind]

    #     if dif[i] > 0:
    #         print("upregulated in TBI area")
    #     else:
    #         print("downregulated in TBI area")
    #     plt.imshow(ion)
    #     plt.scatter(xs, ys, 5.7,c="r", marker="+", alpha=0.5)
    #     plt.show()

def get_tbi_mask(mask):
    # getting mask of ROI based on previously segmented image
    
    _,gray_img = cv2.threshold(mask,0.00001,1,cv2.THRESH_BINARY)
    
    # plt.imshow(gray_img)
    # plt.title('TBI mask')
    # plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    dilate_mask = cv2.dilate(gray_img, kernel, iterations=5)
    # plt.imshow(dilate_mask)
    # plt.title('TBI mask -dilate')
    # plt.show()
    x,y = gray_img.shape
    return gray_img.reshape(x*y), dilate_mask.reshape(x*y)

# def get_mask_contour(mask):
#     _, threshold = cv2.threshold(mask, 0.1, 255, cv2.THRESH_BINARY) 
#     # Detecting contours in image. 
#     contours, _= cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#     xs,ys = [],[]
#     for i in contours:
#         for k in i:
#             x,y = k[0]
#             xs.append(x)
#             ys.append(y)
#     return xs,ys

def get_colocalisation(key, dict_mask, dict_ms, diff_dependent = False, dif = [], show=False):
    # tbi_wt_medians =dict_label_medians[key]
    
    mask, dil_mask= get_tbi_mask(dict_mask[key])
    msn = dict_ms[key]
    # msn = ms.norm(3)

    if diff_dependent:
        filtered_sim_score, mzs, cos_sim_score = get_difdep_colocalised_ions(dil_mask, dict_ms[key],
                                                       filter_score_threshold=0.5, diff_dependent=diff_dependent, dif = dif)
    else:
        filtered_sim_score, mzs, cos_sim_score = get_colocalised_ions(dil_mask, dict_ms[key],
                                                       filter_score_threshold=0.5)
    # print(filtered_sim_score)
    if show:
        xs,ys = get_mask_contour(dict_mask[key])

        for i,score in filtered_sim_score[:10]:
            ion  = msn.int_mat[:,:,i]
            # if(np.abs(tbi_wt_medians[i][0] - tbi_wt_medians[i][1]) > 1.5):
            print(i, msn.mz[i], score)
                
            # msn.plot_ion('ion', 'static', index = [i], title = str(msn.mz[i]))
            plt.imshow(ion)
            plt.scatter(xs, ys, 5.7,c="r", marker="+")
            plt.show()

    return filtered_sim_score, cos_sim_score

def get_inverse_colocalisation(key, dict_mask, dict_ms, dict_label_medians):
    tbi_wt_medians =dict_label_medians[key]
    
    mask, dil_mask= get_tbi_mask(dict_mask[key])
    msn = dict_ms[key]
    # msn = ms.norm(3)
   
  
    
    filtered_sim_score, mzs, cos_sim_score = get_inv_colocalised_ions(dil_mask, dict_ms[key],
                                                       filter_score_threshold=0.5)


    return filtered_sim_score, cos_sim_score
        
# TODO: add contour of ROI on mz image DONE

def plot_roi_ions(msn, median, xs,ys):
     
    p = list(np.abs(median))
    indices, p_sorted = zip(*sorted(enumerate(p), key=itemgetter(1))[::-1])
    #  print(len(p_sorted[p_sorted > 0]))
    p_sorted = p_sorted/np.max(p_sorted)
    print(len(np.nonzero(p_sorted)[0]))
    print('p_sorted',p_sorted[:5])

    for i in indices[:10]:
        if median[i] != 0:
            print(i, msn.mz[i])
            ion = msn.int_mat[:,:,i]
        
            
        
            if median[i] > 0:
                print("upregulated in TBI area")
            else:
                print("downregulated in TBI area")
            plt.imshow(ion)
            plt.scatter(xs, ys, 5.7,c="r", marker="+")
            plt.show()



def match(listofreference, listofvalues, tolerance):
    # matching two mz lists
    # output: mz in common within set range
    count = 0
    matches = []
    for indexi,i in enumerate(listofvalues):
        range = utils.get_mz_tolerance_range(i, tolerance)
        for indexj,j in enumerate(listofreference):
            if j>= range[0] and j<=range[1]:
                count+=1
                matches.append((j,i,indexj, indexi))
    return count, matches



# def get_tbi_mask(sample_imat, index):
#     # getting mask of ROI based on previously segmented image
#     tissue, bkg = sample_imat.create_mask(labels_roi=[index], gmm=True)
#     tbi = np.sum(sample_imat.int_mat_msk.int_mat, axis=2)
#     _,gray_img = cv2.threshold(tbi,0.00001,1,cv2.THRESH_BINARY)
#     # plt.imshow(gray_img)
#     # plt.title('TBI mask')
#     # plt.show()
#     tbi_mask = gray_img.reshape(sample_imat.x *sample_imat.y)
#     return tbi_mask
import cv2
def get_difdep_colocalised_ions(tbi_mask, sample_imat, filter_score_threshold = 0.3, diff_dependent=False, dif = [],cov_mat=[]):
    # getting ions that colocalise with the extracted mask
    # similarity metric used - manhattan distance
    # particulary good for image comparison
    cos_sim_scores = []
    mzs = []
    final_sim_score = []
    for i,spec in enumerate(sample_imat.int_mat_flat.T):
        # sim_score = distance.cityblock(spec, tbi_mask)
        # correlation, braycurtis, cosine, euclidean,sqeuclidean,jensenshannon,
        # chebyshev
        
        spec[np.isnan(spec)] = 0
        spec_scaled = (spec - spec.min()) / (spec.max() - spec.min())
        # print(dif[i]>=0, distance.cosine(tbi_mask,spec_scaled), distance.cosine(tbi_mask,1-spec_scaled),
        #       distance.cosine(1-tbi_mask,spec_scaled), distance.cosine(1-tbi_mask,1-spec_scaled))
        print(dif[i])

        plt.imshow(spec_scaled.reshape(79,92))
        plt.show()

        # Keep top 10% brightest pixels
        threshold = np.percentile(spec_scaled, 90)
        _, spec_scaled = cv2.threshold(spec_scaled, threshold, 255, cv2.THRESH_BINARY)

        # Or manually
        # intense_pixels = np.where(spec_scaled >= threshold, img, 0)

        # threshold = np.percentile(spec_scaled, 99)
        # spec_scaled = np.where(spec_scaled >= threshold, 255, 0).astype(np.uint8)
        plt.imshow(spec_scaled.reshape(79,92))
        plt.show()

                # Otsu's thresholding after Gaussian filtering
        # blur = cv2.GaussianBlur(spec_scaled.astype(np.uint8),(5,5),0)
        # ret3,spec_scaled = cv2.threshold(spec_scaled.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # plt.imshow(spec_scaled)
        # plt.show()
        # print(spec_scaled.shape)
        spec_scaled = spec_scaled.reshape(-1)
        if dif[i] >= 0:
            sim_score = distance.cosine(tbi_mask,spec_scaled)
            print(i,sim_score)
            
        else:
            print('negative')
            sim_score = distance.cosine(tbi_mask, 1-spec_scaled)
            # sim_score = 1-distance.cosine(tbi_mask, spec_scaled)
            # print(i,sim_score, distance.cosine(tbi_mask, spec_scaled))
        cos_sim_scores.append((i, sim_score))
    sorted_cos_sim_scores = sorted(cos_sim_scores, key=lambda x: x[1])
    print(sorted_cos_sim_scores)
    sorted_cos_sim_scores_max = max(sorted_cos_sim_scores,  key=lambda x: x[1])[1]
    sorted_cos_sim_scores_min = min(sorted_cos_sim_scores,  key=lambda x: x[1])[1]
    sorted_cos_sim_scores_norm = [(s[0],(s[1]-sorted_cos_sim_scores_min)/
                                   (sorted_cos_sim_scores_max-sorted_cos_sim_scores_min))
                                   for s in sorted_cos_sim_scores]
    cos_sim_scores_norm = [(s[0],(s[1]-sorted_cos_sim_scores_min)/
                                   (sorted_cos_sim_scores_max-sorted_cos_sim_scores_min))
                                   for s in cos_sim_scores]
    filtered_sim_score = list(filter(lambda x: x[1]<=filter_score_threshold, 
                                     sorted_cos_sim_scores_norm))
 

    return filtered_sim_score, mzs, cos_sim_scores_norm


def get_colocalised_ions(tbi_mask, sample_imat, filter_score_threshold = 0.3,cov_mat=[]):
    from numpy import dot
    from numpy.linalg import norm


    # getting ions that colocalise with the extracted mask
    # similarity metric used - manhattan distance
    # particulary good for image comparison
    cos_sim_scores = []
    mzs = []
    final_sim_score = []
    for i,spec in enumerate(sample_imat.int_mat_flat.T):
        # sim_score = distance.cityblock(spec, tbi_mask)
        # correlation, braycurtis, cosine, euclidean,sqeuclidean,jensenshannon,
        # chebyshev
        
        spec[np.isnan(spec)] = 0
        spec_scaled = (spec - spec.min()) / (spec.max() - spec.min())

        # sim_score = cosine_similarity(np.array(tbi_mask).reshape(-1, 1),
                                    #   np.array(spec_scaled).reshape(-1, 1))
        sim_score = distance.cosine(tbi_mask,spec_scaled)
        # sim_score = dot(tbi_mask, spec_scaled)/(norm(tbi_mask)*norm(spec_scaled))
        # print(sim_score)
        
        cos_sim_scores.append((i, sim_score))
    sorted_cos_sim_scores = sorted(cos_sim_scores, key=lambda x: x[1])
    print(sorted_cos_sim_scores)
    sorted_cos_sim_scores_max = max(sorted_cos_sim_scores,  key=lambda x: x[1])[1]
    sorted_cos_sim_scores_min = min(sorted_cos_sim_scores,  key=lambda x: x[1])[1]
    sorted_cos_sim_scores_norm = [(s[0],(s[1]-sorted_cos_sim_scores_min)/
                                   (sorted_cos_sim_scores_max-sorted_cos_sim_scores_min))
                                   for s in sorted_cos_sim_scores]
    cos_sim_scores_norm = [(s[0],(s[1]-sorted_cos_sim_scores_min)/
                                   (sorted_cos_sim_scores_max-sorted_cos_sim_scores_min))
                                   for s in cos_sim_scores]
    filtered_sim_score = list(filter(lambda x: x[1]<=filter_score_threshold, 
                                     sorted_cos_sim_scores_norm))
 
    for i,score in filtered_sim_score:
        mzs.append(sample_imat.mz[i])
        # print(i)
        # if i in filtered_indices:
        #      final_sim_score.append((i,score))
        #      mzs.append(sample_imat.mz[i])
        # else:
        #     print('no')
        # sample_imat.plot_ion('ion', 'static', index = [i], 
        #                      title = str(round(sample_imat.mz[i],4))+'\n Score: '+str(round(score,2)))
        # plt.show()
    return filtered_sim_score, mzs, cos_sim_scores_norm

def get_inv_colocalised_ions(tbi_mask, sample_imat, filter_score_threshold = 0.3, cov_mat=[]):
    # getting ions that colocalise with the extracted mask
    # similarity metric used - manhattan distance
    # particulary good for image comparison
    cos_sim_scores = []
    mzs = []
    final_sim_score = []
    for i,spec in enumerate(sample_imat.int_mat_flat.T):
        # sim_score = distance.cityblock(spec, tbi_mask)
        # correlation, braycurtis, cosine, euclidean,sqeuclidean,jensenshannon,
        # chebyshev
        
        spec[np.isnan(spec)] = 0
        spec_scaled = (spec - spec.min()) / (spec.max() - spec.min())
        sim_score = distance.cosine(tbi_mask,1-spec_scaled)
        cos_sim_scores.append((i, sim_score))
    sorted_cos_sim_scores = sorted(cos_sim_scores, key=lambda x: x[1])
    print(sorted_cos_sim_scores)
    sorted_cos_sim_scores_max = max(sorted_cos_sim_scores,  key=lambda x: x[1])[1]
    sorted_cos_sim_scores_min = min(sorted_cos_sim_scores,  key=lambda x: x[1])[1]
    sorted_cos_sim_scores_norm = [(s[0],(s[1]-sorted_cos_sim_scores_min)/
                                   (sorted_cos_sim_scores_max-sorted_cos_sim_scores_min))
                                   for s in sorted_cos_sim_scores]
    cos_sim_scores_norm = [(s[0],(s[1]-sorted_cos_sim_scores_min)/
                                   (sorted_cos_sim_scores_max-sorted_cos_sim_scores_min))
                                   for s in cos_sim_scores]
    filtered_sim_score = list(filter(lambda x: x[1]<=filter_score_threshold, 
                                     sorted_cos_sim_scores_norm))
 
    for i,score in filtered_sim_score:
        mzs.append(sample_imat.mz[i])
        # print(i)
        # if i in filtered_indices:
        #      final_sim_score.append((i,score))
        #      mzs.append(sample_imat.mz[i])
        # else:
        #     print('no')
        # sample_imat.plot_ion('ion', 'static', index = [i], 
        #                      title = str(round(sample_imat.mz[i],4))+'\n Score: '+str(round(score,2)))
        # plt.show()
    return filtered_sim_score, mzs, cos_sim_scores_norm

def get_filtered_mz(dict_vals, key_index,val_group, filter_sim_score=0.3):
    keys = list(dict_vals.keys())
    sample = dict_vals[keys[key_index]]
    tbi_mask_sample = get_tbi_mask(sample[0], val_group)
    filtered_sim_score_sample,mzs_sample = get_colocalised_ions(tbi_mask_sample, sample[0], sample[1][0], filter_score_threshold=filter_sim_score) 
    return filtered_sim_score_sample,mzs_sample

def get_intersection_between_matches(match1, match2):
    # e = [e[0] for e in match1]
    # e2 = [e[0] for e in match2]
    # inters = set(e2).intersection(set(e))
    inters = []
    match1 = sorted(match1, key=lambda x: x[0])
    match2 = sorted(match2, key=lambda x: x[0])
    for i in match1:
        for j in match2:
            if i[0] == j[0]:
                inters.append((i,j))
    return inters

def get_difference_between_matches(match1, match2):
    diff = []
    match1 = sorted(match1, key=lambda x: x[0])
    match2 = sorted(match2, key=lambda x: x[0])
    # elem1 = [e[0] for e in match1]
    elem2 = [e[0] for e in match2]
    for i in match1:
        if i[0] not in elem2:
            diff.append((i))
    return diff


def get_difference_between_mz(mzlist, match, tolerance):
    
    diff = []
    mzlist = sorted(mzlist)
    # match = sorted(match)
    match = sorted(match, key=lambda x: x[0])
    elems = [utils.get_mz_tolerance_range(e[0], tolerance) for e in match]
    print(len(mzlist))
    print(len(elems))
    # mzlist = [0.9, 1.3, 1.4, 1.7]
    # elems = [(1.1, 1.5), (1.2,1.6), (2.1,2.2)]

    j = 0
    for index,i in enumerate(mzlist):
        # print(i)
        while j < len(elems):
           
            if i<=elems[j][0]:
                diff.append((i, index))
                # reset to back to 0 if sorted. maybe not
                # j = 0
                break
            
            elif i> elems[j][0]:

                if i > elems[j][1]:
                    j += 1
                    print(j)
                else:
                    j=0
                    break
               
    
    return diff




def get_region_filtered_images(matches,sample, filtered_sim_score_sample, index = 3):

    p = [p[index] for p in matches]
    # for i in p:
    # for i in [3,10,18,24,30]:
        # print(sample[0].mz[filtered_sim_score_sample[i][0]])
        # plt.imshow(sample[0][filtered_sim_score_sample[i][0]])
        # plt.show()

def plot_mz(sample, mz, tol =200):
    ind = sample.get_index(mz, tol)
    print(ind)
    print(sample.mz[ind])
    plt.imshow(sample[ind])
    return sample[ind]