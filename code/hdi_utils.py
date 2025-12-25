import pandas as pd
import numpy as np
import imat


def get_hdi_data(file, flip_image=True, sort_mz=True, header = 2, preproc = False):
    ''' 
        Read in HDI data file.
        Returns sorted intensity matrices and mz list.
    '''
    mat = pd.read_csv(file, sep='\t', header = header, index_col= False)
   
    int_mat = mat.drop(columns = list(mat.columns[0:3])) #get rid of first 3 columns (pixel indices and coordinates)
    
 
    if preproc:
        int_mat = int_mat.drop(columns =list(int_mat.columns[-2:]))
        mz_list = [float(i) for i in list(mat.columns)[3:-2]] 
        xaxis = int(np.nanmax(mat.T.iloc[1]))
        yaxis = int(np.nanmax(mat.T.iloc[2]))
       
        int_mat= int_mat[1:]
        
       

    else:    
        mz_list = [float(i) for i in list(mat.columns)[3:]] #get mz list which is the header
        coord = mat.iloc[:,2] #get x coordinates
        yaxis = int(np.count_nonzero(coord == coord[0])) #count occurences of first element
        xaxis = int(len(coord)/yaxis) #get the shape of the other coordinate
        
        if xaxis*yaxis != len(int_mat):
            print('Missing rows:', len(int_mat)-xaxis*yaxis)
        int_mat = int_mat[:xaxis*yaxis]
    
    new_mat = np.array(int_mat).reshape(xaxis, yaxis, len(mz_list)) # reshape
    # print(new_mat.shape)
    # new_mat = new_mat.T
    # print(new_mat.shape)
    
    
    new_mat = np.rot90(new_mat)

    if flip_image:
        new_mat = np.flip(new_mat, 0) #flip ca-i cu susu-n jos
        new_mat = np.flip(new_mat, 1)

    if sort_mz:
        # sort data by mz values
        axis = new_mat.shape
        data_df = pd.DataFrame(new_mat.reshape(axis[0]*axis[1], axis[2]), columns=mz_list)
        data_df_sorted = data_df.reindex(sorted(data_df.columns), axis=1) 

        data_mat_sorted = np.array(data_df_sorted).reshape(axis[0],axis[1], axis[2])
        data_mz_sorted = np.array(data_df_sorted.columns)

            
        return imat.imat(data_mat_sorted, data_mz_sorted)

    else:
        return imat.imat(new_mat, np.array(mz_list))
    

def filter_mz(pca, imat_dict_unsorted, name_id, loadings_intensity_threshold,
              data_file, filtered_images_folder, filename):

    z=0
    new_p=[]
    new_mz=[]
    ind = []
    for i in range(len(pca[name_id][1])):
        if pca[name_id][1][i]>loadings_intensity_threshold:
            z+=1
            new_p.append(pca[name_id][1][i])
            new_mz.append(imat_dict_unsorted[name_id][0].mz[i])
            ind.append(i+3)
    print(z, 'features remain.')


    # f = pd.read_csv(data_file, sep='\t',header=None, names = range(2006),usecols=[0,1,2]+ind+[2003,2004,2005])
    # f.to_csv(filtered_images_folder+filename, sep='\t', index=False, header='')

def export_files(data_file_path, mz_file_path):

    mat_file = data_file_path
    np.save(mat_file, data_mat_sorted)

    mz_file = mz_file_path
    np.save(mz_file, data_mz_sorted)    




    # def get_pixel_chromatogram(self, x_coord, y_coord):
    #     pixel_z_chromatogram = [self.intensity_matrices[i][x_coord][y_coord] for i in range(self.mass_channels_no)]
    #     return pixel_z_chromatogram

    # def plot_pixel_chromatogram(pixel_z_chromatogram):
   
    #     print('Descriptive stats: min,max,std:',np.min(pixel_z_chromatogram), np.max(pixel_z_chromatogram), np.std(pixel_z_chromatogram))
    #     sns.displot(pixel_z_chromatogram, kind = 'kde', fill=True, legend=False)

    

    #     plt.figure(figsize=(30,10))
    #     plt.stem(data_mz, pixel_z_chromatogram, markerfmt = ' ', linefmt = 'C7-')
    #     plt.xlim(np.min(data_mz), np.max(data_mz))
    #     # plt.ylim(bottom=0)
    #     plt.ylabel('Ion Intensity')
    #     plt.xlabel('m/z')
    #     #plt.ylim(0,12)
        
    #     plt.show()
