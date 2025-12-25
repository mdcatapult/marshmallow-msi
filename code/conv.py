# 'amn'

from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import imat
import hdi_utils




class imat_conv():

    def __init__(self, file_path, format):
        """
        Initialize the imat object.

        Parameters
        ----------
        file_path : str
            the path to the imzML file
        format : str
            the format of the imzML file. Can be 'hdi', 'bruker', or 'waters'

        Returns
        -------
        None

        Notes
        -----
        Depending on the format, this class will either use the hdi_utils or pyimzml to read in the data.
        """
        self.file_path = file_path
        self.f = format
     

        if self.f == 'hdi':
            self.imat = hdi_utils.get_hdi_data(self.file_path )

        else:
            self.parser = ImzMLParser(self.file_path)
            self.spectra = self._get_spectra()
            self.mz = self.spectra[0][0] #because all pixels have the same mz list now after alignment
            self.mzn = len(self.mz)
            self.len = len(self.spectra)

            xcoords = [self.spectra[i][2][0] for i in range(self.len)]
            ycoords = [self.spectra[i][2][1] for i in range(self.len)]

            xmax, xmin = max(xcoords), min(xcoords)
            ymax, ymin = max(ycoords), min(ycoords)

            lenx, leny = xmax-xmin+1, ymax-ymin+1
            im = np.zeros((lenx, leny, self.mzn))

            for i in range(self.mzn):
                for j in range(self.len):
                    intensity = self.spectra[j][1][i]
                    x = self.spectra[j][2][0] - xmin
                    y = self.spectra[j][2][1] - ymin

                    im[x][y][i] = intensity


            if self.f == 'cardinal':
                # after peak picking with waters
                # don't exactly know why though
                im = im.reshape(leny,lenx,self.mzn)

            if self.f == 'bruker':
                im_copy = np.copy(im)
                im_rot = np.rot90(im_copy)
                im_flip = np.flip(im_rot,0)
                # for some samples it's flipped 0, others 1
                im = im_flip

            

            self.imat = imat.imat(im, self.mz)


        
        

        # self.spectra, self.mz, self.all_data = self._get_spectra()

        


        # self.x = max(self.parser.coordinates, key=lambda item: item[1])[1]
        # self.xmin = min(self.parser.coordinates, key=lambda item: item[1])[1]
        # self.y = max(self.parser.coordinates, key=lambda item: item[0])[0]
        # self.ymin = min(self.parser.coordinates, key=lambda item: item[0])[0]

        # if self.f == 'bruker':
           
        #     x = self.y-self.ymin+1
        #     y = self.x-self.xmin+1
          

        # self.x = x
        # self.y = y


        # self.nspec= self.x*self.y*len(self.mz)

        # xy = np.array(self.spectra).shape[1]

        # mzn = np.array(self.spectra).shape[0]
        
        
        # if int(self.nspec) == int(xy*mzn):

        #     image_data = np.array(self.spectra).T.reshape(self.x, self.y, len(self.mz))
                
            
        # else:
        #     print('unmatch')
        #     print(len(np.array(self.spectra).T), self.x, self.y)
        #     print(mzn, len(self.mz))
        #     if (self.x*self.y < xy):
        #         image_data = np.array(self.spectra).T[:self.x*self.y, :]
        #     else:
        #         print(xy, self.x, self.y)
        #         self.x = int(xy/self.y) 
                
        #         image_data = np.array(self.spectra).T[:(self.x*self.y)]
            
        #     print(image_data.shape)
        #     image_data = image_data.reshape(self.x,self.y, len(self.mz))
 

        # if self.f == 'waters':
        #     self.imat = imat.imat(image_data,self.mz)


        # elif self.f == 'bruker':
        #     self.imat = imat.imat(np.flip(np.rot90(image_data),0),self.mz)

       
        


    def _get_spectra(self):

        """
        Reads the spectra data from the imzML file and returns a list of lists, where each sublist contains the mz values, intensity values, and coordinates of a given spectrum.

        Returns
        -------
        list
            a list of lists, where each sublist contains the mz values, intensity values, and coordinates of a given spectrum
        """
        my_spectra = []
       
        # NEEDED for Bruker file until better solution written, i.e. use tghe x,y,z coord to fill in the matrix
        # if self.f == 'bruker':
        #     sort_index = [i for i, x in sorted(enumerate(self.parser.coordinates), key=lambda x: x[1])]
        
        # if self.f == 'bruker':
        #     for idx in sort_index:
        #         mzs, intensities = self.parser.getspectrum(idx)
        #         my_spectra.append([mzs, intensities, self.parser.coordinates[idx]])
                
        # elif self.f == 'waters':
        for idx, (x,y,z) in enumerate(self.parser.coordinates):
            mzs, intensities = self.parser.getspectrum(idx)
            my_spectra.append([mzs, intensities, (x,y,z)])

        # rearranged_spectra = np.array(my_spectra).transpose(1,0)
        # rearranged_spectra = list(zip(*rearranged_spectra[1])) 
        # where the intensity values are stored rearranged_arr[0] where mz values are stored, [2] where the coordinates are stored
        # return rearranged_spectra, np.array(my_spectra[0][0], dtype='float'), my_spectra
        return my_spectra








                


    


