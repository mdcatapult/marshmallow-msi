import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generator
import imat




class synthMSI():

    def __init__(self, no_channels, wt_mask, roi_mask, roi_specific_channels = .5):
        self.no_channels = no_channels
        self.wt_mask = wt_mask
        self.roi_mask = roi_mask
        self.get_labels()

        self.roi_specific_channels = roi_specific_channels
        

    def get_labels(self):
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        dilate_mask = cv2.dilate(self.roi_mask, kernel, iterations=3)
           
        self.roi_labels = np.where(dilate_mask == 255)
        self.wt_labels = np.where(self.wt_mask == 255)
        tbi_wt_mask = np.zeros((self.roi_mask.shape))

        tbi_wt_mask[self.wt_labels] = 1
        tbi_wt_mask[self.roi_labels] = 2
        self.final_labels = np.where(tbi_wt_mask == 1)


    def create_channel(self, tissue_intensity,
                        random_blob_intensity, no_random_blobs, cov_matrix,
                        roi_intensity, spread_factor, subtract=False, nroi=0):
            

        new_ion = np.zeros((self.wt_mask.shape))
       
        new_ion[self.wt_labels] = tissue_intensity
        # new_ion[final_labels] = tissue_intensity
        
        
        labeled_image, count = ski.measure.label(self.roi_mask, return_num=True)
        random_blobs = generator.make_random_blobs(new_ion, self.final_labels,intens=random_blob_intensity, no_blobs=no_random_blobs,
                                                cov_matrix=cov_matrix)
        roi_blobs = generator.get_roi_blobs(new_ion, labeled_image, count, spread_factor, roi_intensity)

        # randomly select one of the blobs in the roi
        if nroi == 1:
            iroi = np.random.randint((len(roi_blobs)))
            # print(iroi)
            if subtract:
                final = new_ion - roi_blobs[iroi]-random_blobs
            else:
                final = new_ion + roi_blobs[iroi]+random_blobs
        # all blobs in the roi
        if nroi == -1:
            for i in roi_blobs:
                if subtract:
                    new_ion -= i
                else:
                    new_ion += i
            # print(roi_blob)
            if subtract:
                final = new_ion -random_blobs
            else:
                final = new_ion +random_blobs

        # no roi
        if nroi == 0:
            final = new_ion + random_blobs
        # plt.imshow(final)
        # plt.show()
        final_poisson = generator.add_poisson_noise(final)

        # plt.imshow(final_poisson)
        return final_poisson


    def generate_dataset(self, range = [700,1800]):
        mz_values = np.linspace(range[0], range[1], self.no_channels)
        self.generate_channels()

        new_imat_array = np.array(self.new_imat_array).swapaxes(0,1).swapaxes(1,2)
        test_ms = imat.imat(new_imat_array, mz_values)
        test_ms.plot_ion('tic', 'static')
        plt.show()
        return test_ms

    def generate_channels(self):
        # Channel generation

        self.new_imat_array = []
        self.labels = []

        n_con_ch = int(self.no_channels*(1-self.roi_specific_channels))
        n_set_roi_ch = int(self.no_channels*self.roi_specific_channels/5)
        
        # generate control mass channels
        for i in range(n_con_ch):
            nrb = np.random.randint(5)
            tissue_int = np.random.randint(5)+1 
            blob_int = (tissue_int)*3
            ion = self.create_channel(
                                tissue_intensity = tissue_int,
                                random_blob_intensity = blob_int, 
                                no_random_blobs = nrb, cov_matrix=(2,2),
                                roi_intensity = 0, spread_factor = 1.5, nroi=0)
            plt.show()
            self.new_imat_array.append(ion)
            self.labels.append(0)

        # generate roi-specific mass channels
        # first condition: upregulation
        for i in range(n_set_roi_ch):
            # establish ratio T:R
            
            # random generation of blob numbers
            nrb = np.random.randint(5)
            tissue_int = np.random.randint(5) 
            blob_int = (tissue_int+1)*3
            roi_int = blob_int*1

            ion = self.create_channel(
                                tissue_intensity = tissue_int,
                                random_blob_intensity = blob_int, 
                                no_random_blobs = nrb, cov_matrix=(2,2),
                                roi_intensity = roi_int, spread_factor = 1,nroi=-1)
            # plt.show()
            self.new_imat_array.append(ion)
            self.labels.append(1)

        # second condition: upregulation: roi, blob > tissue, roi=blob
        for i in range(n_set_roi_ch):
        
            nrb = np.random.randint(5)
            tissue_int = np.random.randint(5) 
            roi_int = (tissue_int+1)*5
            blob_int = (tissue_int+1)*5

            ion = self.create_channel(
                                tissue_intensity = tissue_int,
                                random_blob_intensity = blob_int, 
                                no_random_blobs = nrb, cov_matrix=(2,2),
                                roi_intensity = roi_int, spread_factor = 1, nroi=-1)

            self.new_imat_array.append(ion)
            self.labels.append(1)

        # third condition: upregulation: roi, blob < tissue, roi=blob
        for i in range(n_set_roi_ch):
      
            nrb = np.random.randint(5)
            # ensure > 0
            tissue_int = np.random.randint(5) 
            
            blob_int = (tissue_int+1)*5
            roi_int = (tissue_int+1)*10

            ion = self.create_channel(
                                tissue_intensity = tissue_int,
                                random_blob_intensity = blob_int, 
                                no_random_blobs = nrb, cov_matrix=(2,2),
                                roi_intensity = roi_int, spread_factor = 1.5, nroi=1, subtract=False)
            self.new_imat_array.append(ion)
            self.labels.append(1)

        # fourth condition: downregulation: roi, blob < tissue, roi=blob
        for i in range(n_set_roi_ch):
        
            nrb = np.random.randint(5)

            # ensure > 0
            tissue_int = np.random.randint(5)+1
            roi_int = (tissue_int)*1.4
            blob_int = (tissue_int)*0
            ion = self.create_channel(
                                tissue_intensity = tissue_int,
                                random_blob_intensity = blob_int, 
                                no_random_blobs = nrb, cov_matrix=(2,2),
                                roi_intensity = roi_int, spread_factor = 1.5, nroi=-1, 
                                subtract=True)
            
            self.new_imat_array.append(ion)
            self.labels.append(2)

        # fifth condition: downregulation: roi, blob < tissue, roi=blob
        for i in range(n_set_roi_ch):
        
            nrb = np.random.randint(5)
            tissue_int = np.random.randint(9)+1 
            roi_int = (tissue_int)*0.4
            blob_int = (tissue_int)*0.4

            ion = self.create_channel(
                                tissue_intensity = tissue_int,
                                random_blob_intensity = blob_int, 
                                no_random_blobs = nrb, cov_matrix=(2,2),
                                roi_intensity = roi_int, spread_factor = 1.5, nroi=-1, 
                                subtract=True)
            
            
            self.new_imat_array.append(ion)
            self.labels.append(2)



    

      

    

    



