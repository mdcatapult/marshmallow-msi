import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plotly.subplots import make_subplots


from scipy import ndimage as ndi

from pyimzml.ImzMLWriter import ImzMLWriter
from skimage.filters import threshold_otsu
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture
import cv2
import networkx as nx
from faiss_imputer import FaissImputer
import matplotlib
from skimage import exposure
import utils

class imat():

    def __init__(self,intensity_matrices,mass_channels_list):
        """
        Initialize the object with intensity matrices and mass channels list.
        Args:
            intensity_matrices (numpy.ndarray): A 3D array where the first two dimensions 
                                                represent spatial coordinates and the third 
                                                dimension represents intensity values for 
                                                different mass channels.
            mass_channels_list (list): A list of mass channels corresponding to the third 
                                       dimension of the intensity_matrices.
        Attributes:
            int_mat (numpy.ndarray): The input intensity matrices.
            x (int): The size of the first spatial dimension.
            y (int): The size of the second spatial dimension.
            mz (list): The input list of mass channels.
            mzn (int): The number of mass channels.
            ids (list): A list of identifiers for each mass channel.
            int_mat_flat (numpy.ndarray): The flattened intensity matrix with shape 
                                          (x * y, mzn).
            tot_ion_int_mat (numpy.ndarray): The total ion intensity matrix with shape 
                                             (x, y).
            tot_pix_int_mat (numpy.ndarray): The total pixel intensity matrix.
        """
        
        self.int_mat = intensity_matrices
        self.x = intensity_matrices.shape[0]
        self.y = intensity_matrices.shape[1]
        self.mz = mass_channels_list
        self.pixels = self.x*self.y

        self.mzn = len(self.mz)
        self.ids = ['PK'+str(i) for i in range(self.mzn)]
        
        self.int_mat_flat = self.int_mat.reshape(self.x * self.y, self.mzn)
        self.tot_ion_int_mat = self._sum_ions(self.int_mat_flat).reshape(self.x, self.y)
        self.tot_pix_int_mat = self._sum_pixels(self.int_mat_flat)
        # add mean intensity per mz
        self.mean_ints = [np.mean(self.int_mat_flat[:,i]) for i in range(self.mzn)]
        self.top_mz_ind = sorted(range(len(self.mean_ints)),key=self.mean_ints.__getitem__, reverse=True)


    def plot_top_ions(self,n,cscale='viridis', savefig=False, location='',**kwargs):
        for index,i in enumerate(self.top_mz_ind[:n]):
            print(self.mz[i])
            self.plot_ion('ion', 'static', index=[i], cscale=cscale, **kwargs)
            
            if savefig:
                da = exposure.equalize_hist(self.int_mat[:,:,i])
                masked_array = np.ma.array(da, mask=np.isnan(da))
                masked_array = masked_array/np.max(masked_array)*255
                cmap = matplotlib.cm.viridis
                cmap.set_bad('black',0.)
                plt.imsave(location+'top'+str(index)+'.png',masked_array.astype('uint8'), cmap=cmap)
                
            plt.show()


    #####--------------------Class special methods--------------------####### 
    def __repr__(self):
        """
        Returns:
            str: A string representation of the imat object.
        """
        return f'imat object of shape ({self.x}, {self.y}) with {self.mzn} mass channels.'

    def __len__(self):
        """
        Returns:
            int: The number of mass channels of an imat object.
        """
        return self.mzn
    
    def __shape__(self):
        """
        Returns:
            tuple: A tuple containing three elements (self.x, self.y, self.mzn) 
            representing the dimensions of the imat object.
        """
        return (self.x, self.y, self.mzn)
    
    def __getitem__(self, index):
        """
        Parameters:
        index (int or tuple of ints): The index or indices to retrieve from the internal matrix.

        Returns:
        list of numpy.ndarray intensity matrices: The selected elements from the internal matrix. 
        If a single index is provided, a single numpy.ndarray is returned. 
        If multiple indices are provided, a list of numpy.ndarray is returned.
        """
        
        if len(index) > 1:
            return [self.int_mat[:,:,i] for i in index]
        elif len([index]) == 1:
            return self.int_mat[:,:,index]
        else:
            raise ValueError("No index value")

    def __iter__(self):
        """
        Returns an iterator that yields the internal matrix and mz values.
        """
        yield self.int_mat
        # yield self.mz
    

    #####--------------------Class helper and private methods--------------------####### 
    def _sum_ions(self, data_flattened):
        """
        Returns:
        numpy.ndarray: A 1D array containing the sum of ions for each row.
        """
        return np.sum(data_flattened, axis=1)
    
    def _sum_pixels(self, data_flattened):
        """
        Returns:
        numpy.ndarray: The sum of the pixel values along the specified axis.
        """
        return np.sum(data_flattened, axis=0)
    

    #####--------------------Getters and setters methods--------------------####### 
    def get_spectra(self,index):
        """
        Retrieve spectra data based on provided index.

        Returns:
        list or ndarray: If the length of the index is greater than 1, returns a list of spectra data.
                         If the length of the index is 1, returns the spectra data for the given index.
        """
        # print(len(index))
        if len(index) > 1:
            return [self.int_mat_flat[x,:] for x in index]
        elif len(index) == 1:
            if len(index[0].shape) == 2:
                x,y = index
                return self.int_mat[x,y, self.mz]
            else:
                return self.int_mat_flat[index, self.mz]

    def set_labels(self, b_labels, t_labels):
        # in case segmentation has been previously performed
        """
        Set the background and tissue labels for the image.

        Parameters:
        b_labels (array-like): Array containing the background labels.
        t_labels (array-like): Array containing the tissue labels.
        """

        self.bkg_labels = b_labels
        self.tissue_labels = t_labels



    def get_index(self, mz,tol):
        """
        Retrieve the index of a specific m/z value within a given tolerance.

        Parameters:
        mz (float): The m/z value to search for.
        tol (float): The tolerance within which to search for the m/z value.

        Returns:
        int: The index of the m/z value within the specified tolerance.
        """
        ind = utils.get_index_for_specific_mz(tol, mz, np.array(self.mz))
        return ind


#####--------------------Save methods--------------------#######  
    def write_to_imzml(self, output_filepath, mode = 'positive'):
        # using pyimzml package
        # saving just the tissue area, the rest will be set to nan
        """
        Write the tissue area to an imzML file.

        Parameters:
        output_filepath (str): The path to which the imzML file should be written.
        mode (str): The ion mode of the imzML file. Default is 'positive'.

        Notes:
        The background area is set to nan and will be excluded from the output file.
        """
        data = self.int_mat_flat.copy()
        data[self.bkg_labels] = np.nan
        data = data.reshape(self.x, self.y,self.mzn)

        with ImzMLWriter(output_filepath, polarity=mode, mode = 'processed', spec_type='centroid') as w:
            for x in range(self.x):
                for y in range(self.y):
                    if not np.isnan(data[x,y,0]):
                        w.addSpectrum(self.mz, self.int_mat[x,y,:], (x+1,y+1,1))
                
    def write_to_imzmlall(self, output_filepath, mode = 'positive'):
        # saving with background
        """
        Write the entire image (including background) to an imzML file.

        Parameters:
        output_filepath (str): The path to which the imzML file should be written.
        mode (str): The ion mode of the imzML file. Default is 'positive'.

        Notes:
        The background area is included in the output file.
        """
        data = self.int_mat_flat.copy()
        # data[self.bkg_labels] = np.nan
        data = data.reshape(self.x, self.y,self.mzn)
        with ImzMLWriter(output_filepath, polarity=mode, mode = 'processed', spec_type='centroid') as w:
            for x in range(self.x):
                for y in range(self.y):
    
                        w.addSpectrum(self.mz, data[x,y,:], (x+1,y+1,1))
  
    
        
#####--------------------Normalisation methods--------------------#######  
    def norm(self, type):
        """
        Parameters:
        type (int): The type of normalization to apply.
            - 0: TIC normalization
            - 1: Log transformation
            - 2: TIC normalization followed by log transformation
            - 3: Mean normalization followed by log transformation
        """
        if type == 0:
            n = self.tic_normalise()
            return n
        if type == 1: 
            n = self.log_transform()
            return n
        if type == 2:
            n = self.tic_normalise()
            n = self.norm_int_mat.log_transform()
            return n
        if type == 3:
            n = self.mean_normalise()
            n = self.norm_int_mat.log_transform()
            return n
        if type == 4:
            n = self.median_normalise()
            n = self.norm_int_mat.log_transform()
            return n
        if type == 5:
            n = self.median_normalise()
            n = self.norm_int_mat.sqrt_transform()
            return n
        
    def tic_normalise(self):

        """
        Total Ion Count (TIC) normalization.

        Parameters:
        None

        Returns:
        norm_int_mat (imat): The normalized intensity matrix.
        """
        norm_int_mat = np.divide(self.int_mat_flat,
                  np.sum(self.int_mat_flat, axis = 0))
        norm_int_mat = self.ravel_matrix(norm_int_mat)
        self.norm_int_mat = imat(norm_int_mat, self.mz)
        return self.norm_int_mat
    
    def mean_normalise(self):
        """
        Mean normalization.

        Parameters:
        None

        Returns:
        norm_int_mat (imat): The normalized intensity matrix.
        """
        norm_int_mat = np.divide(self.int_mat_flat,
                  np.nanmean(self.int_mat_flat, axis = 0))
     
        norm_int_mat = self.ravel_matrix(norm_int_mat)
        self.norm_int_mat = imat(norm_int_mat, self.mz)
        return self.norm_int_mat
    
    def median_normalise(self):
        """
        Median normalization of the intensity matrix.

        This method normalizes the flattened intensity matrix by dividing each 
        element by the median value of its corresponding mass channel, adding a 
        small constant to avoid division by zero. The normalized matrix is then 
        reshaped and stored.

        Returns:
        norm_int_mat (imat): The normalized intensity matrix.
        """

        norm_int_mat = np.divide(self.int_mat_flat,
                  (np.nanmedian(self.int_mat_flat, axis=0)+0.0000000000000001))
  
        norm_int_mat = self.ravel_matrix(norm_int_mat)
        self.norm_int_mat = imat(norm_int_mat, self.mz)
        return self.norm_int_mat
        
    def log_transform(self):
        """
        Logarithmic transformation of the intensity matrix.

        This method transforms the intensity matrix by applying a natural 
        logarithm with base 2 to each element, adding a small constant to avoid 
        division by zero. The transformed matrix is then stored.

        Returns:
        log_int_mat (imat): The transformed intensity matrix.
        """
        
        self.log_int_mat = imat(np.log2(self.int_mat+1), self.mz)
        return self.log_int_mat
    
    def sqrt_transform(self):
        """
        Square root transformation of the intensity matrix.

        This method transforms the intensity matrix by applying a square root to
        each element, which can help to reduce the effect of extreme values and
        make the data more suitable for visualisation and analysis.

        Returns:
        sqrt_int_mat (imat): The transformed intensity matrix.
        """
        self.sqrt_int_mat = imat(np.sqrt(self.int_mat), self.mz)
        return self.sqrt_int_mat


   
#####--------------------Plotting methods--------------------#######  
    def plot_ion(self, type, mode, index=[], title = '', axis = True, mask=True,hotspot_removal = False, 
                 median_filter = False, hist_eq=False, median_filter_size = 2, savefig = False, location = '', rangecolor=[], cscale = 'viridis',roi=False, roi_loc = [], **kwargs):
        """
        Plots ion images based on the specified type and mode.

        This method visualizes ion images using different plotting modes and types of ions. It
        supports options like hotspot removal, median filtering, and saving the image to a file.

        Parameters:
        type (str): The type of ion plot. Options include 'tic' for Total Ion Current, 'ion' for
                    a single ion, and 'ions' for a selection of ions.
        mode (str): The mode of plotting. Options include 'interactive' for interactive plots
                    and 'static' for static plots.
        index (list, optional): Indices of ions to be plotted, relevant when type is 'ion' or 'ions'.
                                Defaults to an empty list.
        title (str, optional): The title for the plot. Defaults to an empty string.
        axis (bool, optional): Whether to display axes in the plot. Defaults to True.
        mask (bool, optional): Whether to apply a mask to the image. Defaults to True.
        hotspot_removal (bool, optional): Whether to remove hotspots from the image. Defaults to False.
        median_filter (bool, optional): Whether to apply a median filter to the image. Defaults to False.
        savefig (bool, optional): Whether to save the figure to a file. Defaults to False.
        location (str, optional): The file path to save the figure when savefig is True. Defaults to an empty string.
        rangecolor (list, optional): Color range for the plot. Defaults to an empty list.
        **kwargs: Additional keyword arguments for hotspot removal.

        Returns:
        None
        """

        if type == 'tic':
            title = 'TIC '
            imat = self.tot_ion_int_mat
        if type == 'ion':
            title = ' '+title
            imat = self[index]
        if type == 'ions':
            title = 'Selection ions '+title
            imat = np.sum(self[index], axis=0)
            print(imat.shape)
    
        if hotspot_removal:
            imat = self.hot_spot_removal(imat, **kwargs)
            # imat = self._impute_dead_pixel(imat)
            # print('getr', np.where(imat==0))
        if median_filter:
            imat = self.median(imat, size = median_filter_size)
        if mode == 'interactive':
            fig = self._plot_image_interactive(imat)
            fig.update_layout(title=title)
        elif mode == 'static':
            fig = self._plot_image_static(imat,hist_eq, axis=axis, cscale = cscale)
            fig.suptitle(title)
        fig.show()
        imat_flat= imat.reshape(self.x*self.y)
        if hasattr(self, 'bkg_labels'):
                try:
                    
                    imat_flat[self.bkg_labels] = np.nan
                except NameError:
                    pass
        if savefig:
            import matplotlib
            # if len(imat.shape) == 3:
            imat = imat_flat.reshape(self.x, self.y)
            if hist_eq:
                imat_ = np.nan_to_num(imat, nan=0.0) 
                imat_ = exposure.equalize_hist(imat_)
            # plt.imshow(equalized, cmap=cscale)
            # plt.show()
            fig = plt.figure()
            masked_array = np.ma.array(imat_, mask=np.isnan(imat))
            cmap = matplotlib.cm.viridis
            cmap.set_bad('black',0.)
           
            if roi:
                fig, ax = plt.subplots(figsize=(8, 8))

                # Display the array
                ax.imshow(masked_array, cmap='viridis')

                colors = ['red', 'blue', 'yellow', 'green', 'cyan', 'magenta']
                sizes = [60, 30]

                # Add all scatter plots in a loop
                for i, (sx, sy) in enumerate(roi_loc):
                    # s=50*(i+1)
                    # print(s)
                    ax.scatter(sx, sy, 
                            c=colors[i % len(colors)], 
                            s=sizes[i % len(sizes)], 
                            marker='o',
                            edgecolors='white', 
                            linewidths=2, 
                            alpha=0.8,
                            )
                ax.axis('off')
                # Convert figure to array
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.imsave(location, image_array, cmap=cmap,format = 'tiff')
                plt.close(fig)
            
            else:
                plt.imsave(location, masked_array, cmap=cmap,format = 'tiff')
                plt.close(fig)
       

    def plot_spectra(self, type, color = 'blue',index=[]):
        """
        Plot spectra from the given imat data.

        Parameters
        ----------
        type : str
            'total' plots the total spectrum
            'spectrum' plots the spectra from the given index
            'spectra' plots the average spectra from the given index
        color : str
            color of the spectra plot
        index : list or int
            the index or indices of the spectra to plot

        Returns
        -------
        spec : ndarray
            the spectra data
        """
        if type == 'total':
            spec = self.tot_pix_int_mat
        if type == 'spectrum':
            spec = self.get_spectra(index)
        if type == 'spectra':
            # print(self.get_spectra(index))
            spec = np.sum(self.get_spectra(index), axis=0)/len(index)
        fig = self._plot_spectrum(self.mz, spec, color=color)
        fig.show()
        return spec

    def _plot_image_interactive(self, imat, title='', figshow = False, hotspot_removal = False, median_filter = False, savefig = False, location = '', rangecolor=[],cscale = 'viridis'):
        """
        Plot a heatmap of the given imat data using plotly.express.imshow.

        Parameters
        ----------
        imat : 2D array
            The data to plot.
        title : str
            The title of the plot.
        figshow : bool
            If True, show the plot.
        hotspot_removal : bool
            If True, remove the hotspots from the data before plotting.
        median_filter : bool
            If True, apply a median filter to the data before plotting.
        savefig : bool
            If True, save the plot to a file.
        location : str
            The location to save the plot to.
        rangecolor : list
            The range of colors to use in the heatmap. If empty, use the default
            range of [0, 1].

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The plotly figure object.
        """
        import plotly.express as px

        imat = exposure.equalize_hist(imat)
        if len(rangecolor)>0:
            fig = px.imshow(imat, color_continuous_scale=cscale, aspect='equal', range_color=rangecolor)
        else:
            fig = px.imshow(imat,color_continuous_scale=cscale, aspect='equal')
        fig.update_layout(width = 500, height =400, 
                        margin=dict(l=0, r=0, b=0, t=40)
                        )
        fig.update_xaxes(showticklabels=False, visible=False)
        fig.update_yaxes(showticklabels=False, visible=False)
        if figshow:
            fig.show()
        return fig
    
    def _plot_image_static(self, imat, hist_eq, axis=True, cscale = 'viridis'):
        """
        Plot a heatmap of the given imat data using matplotlib.pyplot.imshow.

        Parameters
        ----------
        imat : 2D array
            The data to plot.
        axis : bool
            If True, show the axis.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """
        # from sklearn.preprocessing import MinMaxScaler
        import matplotlib.pyplot as plt
        # from matplotlib.colors import LogNorm, PowerNorm, TwoSlopeNorm
       
        # plt.imshow(imat, cmap=cscale)
        # plt.show()

        if hist_eq:
            imat = exposure.equalize_hist(imat)
        plt.imshow(imat, cmap=cscale)
        # plt.show()
       

        


        # plt.imshow(imat, cmap='viridis')
        # plt.show()
    
        
        # print(np.max(imat), np.min(imat))
        if not axis:
            plt.axis('off')
        fig = plt.gcf()
        # if savefig:
        #     plt.imsave(pathname, imat)
        return fig



    def remove_bimodal_noise(self, img, window_size=3, sensitivity=2):
        """
        Remove scattered noise from bimodal image data
        sensitivity: lower = more aggressive (try 1.5-3)
        """
        from scipy import ndimage
        result = img.copy().astype(float)
        
        # Local median
        local_median = ndimage.median_filter(img, size=window_size)
        
        # Local MAD (Median Absolute Deviation)
        local_dev = ndimage.median_filter(np.abs(img - local_median), size=window_size)
        
        # Replace outliers
        threshold = sensitivity * (local_dev + 1)
        outlier_mask = np.abs(img - local_median) > threshold
        result[outlier_mask] = local_median[outlier_mask]
        
        return result.astype(img.dtype)
    def _selective_median_filter(self,img, threshold=30):
        """
        Only applies median filter to pixels that differ significantly from neighbors
        """
        result = img.copy()
        median = cv2.medianBlur(img, 3)
        
        # Calculate difference between original and median
        diff = np.abs(img.astype(int) - median.astype(int))
        
        # Only replace pixels with large differences (likely noise)
        mask = diff > threshold
        result[mask] = median[mask]
        
        return result   
    def hot_spot_removal(self,im, q=99., **kwargs):
        #from 'palmer'
        """
        Remove hotspots from the image data.

        Parameters
        ----------
        im : 2D array
            The image data.
        q : float
            The quantile to use for thresholding. All values above the qth
            percentile will be set to the qth percentile.
        **kwargs :
            Additional keyword arguments to pass to _impute_dead_pixel.

        Returns
        -------
        im : 2D array
            The image data with hotspots removed.
        """
        import numpy as np
        xic_q = np.percentile(im, q)
        im[im > xic_q] = xic_q
        im = self._impute_dead_pixel(im, **kwargs)
        return im
    
    #     # 
    # def quantile_threshold(im, q_val, notnull_mask=None):
    #     """
    #     Set all values greater than the :code:`q_val`-th percentile to the :code:`q_val`-th percentile (i.e. flatten out
    #     everything greater than the :code:`q_val`-th percentile). For determining the percentile, only nonzero pixels are
    #     taken into account, that is :code:`im[notnull_mask]`.

    #     :param im: the array to remove the hotspots from
    #     :param q_val: percentile to use
    #     :param notnull_mask: index array for the values greater than zero. If None, no mask is used
    #     :return: The :code:`q_val`-th percentile
    #     """
    #     notnull_mask = np.ones(np.shape(im)) if notnull_mask is None else notnull_mask
    #     im_q = np.percentile(im[notnull_mask], q_val)
    #     im_rep = im > im_q
    #     im[im_rep] = im_q
    #     return im_q

    
    def _impute_dead_pixel(self, im, impute = True):
        """
        Impute dead pixels in the given image data.

        Parameters
        ----------
        im : 2D array
            The image data.
        impute : bool
            If True, impute dead pixels using the FaissImputer.

        Returns
        -------
        im : 2D array
            The image data with dead pixels imputed.
        """
        if impute:
            c = im.copy()
            c[c==0] = np.nan
            imputer = FaissImputer(5, strategy='median')
            print('imputer created')
            # print(c.shape)
            if len(c.shape) > 2:
                c = c.reshape(c.shape[0], -1)
            imputer.fit(c)
            c_imputed = imputer.transform(c)
            # most of the values are 0 so not very useful...it will still result in nan values
            c_imputed[np.isnan(c_imputed)] = 0

            # c_imputed = pd.DataFrame(imputer.transform(c)).fillna(0)
            # c_imputed.fillna(0)
            
            # labels = np.where(im == 0)
            # c = im.copy()
            # network_array = nx.grid_2d_graph(*im.shape)
            # len_labels = len(labels[0])
            # # print(len_labels)
            # # impute dead pixels with the mean of the adjacent pixels (this also takes a looong time see nx time complexity)
            # for l in range(len_labels):
            #     neighbours = list(network_array.neighbors((labels[0][l], labels[1][l])))
            #     values = c[neighbours]
            #     values[values==0] = np.nan
            #     mean = np.nanmean(values)
            #     c[labels[0][l]][labels[1][l]] = mean
                
        
            # minval = np.min(im[np.nonzero(im)])
        
            # c.sort()
            # minval = np.unique(c)[35]
            # print('vdvf', minval)
            # im[labels] = minval

            #  impute dead pixels using KNN imputer (takes veeeeery long for 1 dataset, works well with just one 2d array)
            # from sklearn.impute import KNNImputer
        
            # c = im.copy()
            # c[c==0] = np.nan

            # imputer = KNNImputer()
            # # fit on the dataset
            # imputer.fit(c)
            # # transform the dataset
            # Xtrans = imputer.transform(c)
            # print('imputing dead pixels done')
            return c_imputed
        return im

    def median(self,im, **kwargs):
        """
        Apply a median filter to the input image.

        Parameters:
        im (ndarray): The input image array to be filtered.
        **kwargs: Additional keyword arguments to be passed to the median filter function.

        Returns:
        ndarray: The image after applying the median filter.
        """

        from scipy import ndimage
        im = ndimage.filters.median_filter(im,**kwargs)
        # from skimage import filters
        # im = filters.median(im)
        return im
    
    def _gaussian(self, im):
        """
        Apply a Gaussian filter to the input image.

        Parameters:
        im (ndarray): The input image array to be filtered.

        Returns:
        ndarray: The image after applying the Gaussian filter.

        Notes:
        The Gaussian filter is applied with a fixed sigma of 1.
        """
        from scipy.ndimage import gaussian_filter
        im = gaussian_filter(im, sigma=1)
        return im
    
    def _log(self,im):
        """
        Apply a logarithmic transformation to the input image.

        Parameters
        ----------
        im : ndarray
            The input image array.

        Returns
        -------
        ndarray
            The image after applying the logarithmic transformation, with a small
            constant added to avoid log(0).
        """

        import numpy as np
        return np.log(im+1)
    
    def _create_stem_data(self,x,y, baseline=0.):
        '''makes y data passing 0 before inbetween actual value to create data for a stem plot
        x,y are 3 times the original length
        '''
        x=np.repeat(x,3)
        y=np.repeat(y,3)
        y[::3]=y[2::3]=baseline
        return x,y

    def _plot_spectrum(self,x, y, showfig = False, color = 'blue'): 
        """
        Plot a mass spectrum using plotly.

        Parameters
        ----------
        x : array_like
            The m/z values for the spectrum.
        y : array_like
            The intensity values for the spectrum.
        showfig : bool, optional
            Whether to show the figure. Defaults to False.
        color : str, optional
            The color for the spectrum. Defaults to 'blue'.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object.

        Notes
        -----
        The y-axis is set to have a range of zero to the maximum intensity, with a zero line.
        The x-axis is set to have a range of the minimum to maximum m/z value.
        """
        import plotly.graph_objects as go
        x,y = self._create_stem_data(x,y)
        traces = []
        traces.append( [ go.Scatter(x=x, y=y, # plot spectra
                            mode='lines', line_color=color, opacity=0.8,
                            showlegend = False, hoverinfo='none'),
                        go.Scatter(x=x, y=y,         
                            mode='markers',marker_color= color, opacity = 0.6,showlegend = False
                            )]  
                            
                        )
        fig = go.Figure()
        fig.add_traces([traces[0][0], traces[0][1]])
        fig.update_layout(title='Mass Spectra: ', xaxis_title = 'm/z (Da)', 
                        yaxis_title = 'Intensity',
                        yaxis_zeroline=True, xaxis_zeroline=False, 
                        xaxis_range=[np.min(x),np.max(x)],
                        )
        # fig.update_yaxes(type='log')
        if showfig:
            fig.show()
        return fig
    

#####--------------------Segmentation methods--------------------#######  
    def segment(self, n_pc, n_clusters,**kwargs):

        """
        Segment the data using PCA and KMeans clustering.

        Parameters:
        n_pc (int): Number of principal components to use for PCA.
        n_clusters (int): Number of clusters to form using KMeans.
        **kwargs: Additional keyword arguments to be passed to the KMeans algorithm.

        This function first runs PCA on the data to reduce its dimensionality 
        and then applies KMeans clustering to segment the data into clusters.
        """

        self.run_pca(n_pc)
        self.run_kmeans(n_clusters=n_clusters, **kwargs)

    def create_mask(self, labels_roi, gmm=False):
        """
        Create a mask for the given ROI labels.

        Parameters
        ----------
        labels_roi : array_like
            The ROI labels to use for creating the mask.
        gmm : bool, optional
            Whether to use GMM for creating the mask. Defaults to False.

        Returns
        -------
        int_mat_msk, int_mat_bkg : tuple of ndarrays
            The masked and background images.

        Notes
        -----
        The mask is created by thresholding the ROI labels to obtain a binary mask.
        The masked and background images are then created by multiplying the original image with the binary mask.
        """
        self.create_binary_mask(labels_roi, gmm=gmm)
        self.create_masked_tissue()
        return self.int_mat_msk, self.int_mat_bkg

    def show_image_select_ions(self, mz_list_ind, title = 'ind ion image'):

        """
        Show the image of the ions selected by index.

        Parameters
        ----------
        mz_list_ind : array_like
            The indices of the ions to select.
        title : str, optional
            The title of the plot. Defaults to 'ind ion image'.

        Returns
        -------
        None

        Notes
        -----
        The image is created by summing the intensity of the selected ions over the entire image.
        The resulting image is displayed using matplotlib's imshow function.
        """
        sum = np.zeros((self.x,self.y))
        for mz_ind in mz_list_ind:
            print(mz_ind)
            sum += self.int_mat[:,:,mz_ind]

        plt.figure(figsize=(5,5))
        if title:
            plt.title(title)
        else:
            plt.title('Total ion image')
    
        plt.imshow(sum, interpolation = 'none' ,cmap = 'viridis')
        plt.show()

   

    def apply_inter_scaling(self, scx):
        """
        Apply internal scaling to the intensity matrix.

        This method scales the intensity matrix by dividing each element by the 
        corresponding scaling factors provided in `scx`. The resulting matrix is 
        stored as `internorm_int_mat`.

        Parameters
        ----------
        scx : array_like
            The scaling factors used for normalization. Must be broadcastable 
            to the shape of `int_mat`.

        Returns
        -------
        None
        """

        
        self.internorm_int_mat = imat(np.divide(self.int_mat, scx), self.mz)  
        
    def ravel_matrix(self, data_flattened):

        """
        Reshape a flattened data array into the original 3D intensity matrix shape.

        Parameters
        ----------
        data_flattened : numpy.ndarray
            A 2D array with shape (x * y, mzn) representing the flattened intensity matrix.

        Returns
        -------
        numpy.ndarray
            A 3D array with shape (x, y, mzn) representing the original intensity matrix.
        """

        return data_flattened.reshape(self.x, self.y, self.mzn)

    def show_image_otsu_thresh(self):

        """
        Display a total ion image with Otsu thresholding.

        This function applies median filtering and hot spot removal to the total ion intensity matrix. 
        It then calculates an Otsu threshold to identify areas of low intensity. The resulting image 
        is displayed with red markers indicating positions below the threshold.

        Parameters:
        None

        Returns:
        None
        """

        plt.figure(figsize=(5,5))
        # blur = cv2.GaussianBlur(self.tot_ion_int_mat,(7,7),0)
        # blur = ndi.gaussian_filter(self.tot_ion_int_mat, 12)
        blur = median(self.tot_ion_int_mat, size=7)
        smoothed_data_mat = hot_spot_removal(blur, 95)
        thresh = threshold_otsu(smoothed_data_mat.reshape(self.x*self.y))
        
        ssum = self.tot_ion_int_mat
        nrows, ncols = np.where(ssum<=thresh)
        plt.title('Total ions image + Otsu threshold')
        plt.imshow(ssum, cmap = 'viridis')
        plt.scatter(ncols, nrows, 0.6,c="r", marker="+")
        plt.show()
         


    def run_pca(self, n_components):
        """
        Perform Principal Component Analysis (PCA) on the intensity matrix.
        This method preprocesses the intensity matrix by filling NaN values with zero,
        applying a median filter, and removing hot spots. It then performs PCA to reduce
        the dimensionality of the data.
        Args:
            n_components (int): The number of principal components to compute.
        Returns:
            PCA: The fitted PCA object.
        """

        int_mat_flat = self.int_mat_flat.copy()
        data_frame = pd.DataFrame(int_mat_flat)
       
        
        data_frame = data_frame.fillna(0)
        data_frame = data_frame[np.isfinite(data_frame).all(1)]
        data_frame = self.hot_spot_removal(data_frame, 99)
        
        # data_frame = median(data_frame, size=3)
        data_frame = self.median(data_frame,size=2) 
        # data_frame = self._gaussian(data_frame)
         
        pca = PCA(n_components=n_components)
        
        pca.fit(data_frame)
        self.pca_scores = pca.transform(data_frame)

        return pca
    
    def run_nmf(self, n_components):

        """
        Perform Non-negative Matrix Factorization (NMF) on the intensity matrix.
        
        This method applies NMF to the preprocessed intensity matrix to reduce its
        dimensionality. The number of components is specified by the user.
        
        Parameters
        ----------
        n_components : int
            The number of components to compute.
        
        Returns
        -------
        None
        """
        model = NMF(n_components=n_components)
        self.W = model.fit_transform(self.int_mat_flat)
        self.H = model.components_


    def plt_scree_kaiser(self):

        """
        Generates a scree plot of eigenvalues from PCA and applies the Kaiser criterion.

        This method performs PCA with a specified number of components (default 10)
        and plots the eigenvalues to visualize the data variance explained by each 
        principal component. It also plots the explained variance ratio for each component.
        The Kaiser criterion is applied by drawing a reference line at y=1, and the method 
        identifies the number of components with eigenvalues greater than 1, which is 
        returned as the index.

        Returns:
            int: The index of the last principal component with an eigenvalue greater than 1.
        """

        pca = self.run_pca(10) 
        eigenvalues = pca.explained_variance_
        plt.figure(figsize=(3,3))
        plt.rcParams.update({'font.size': 10})
        plt.plot(np.arange(1, len(eigenvalues)+1), eigenvalues, marker='o')
        plt.plot(np.arange(1, pca.n_components_+1), pca.explained_variance_ratio_, 'o-', linewidth=2)
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.title('Scree Plot for Eigenvalues + Kaiser')
        plt.axhline(y=1, color='r', linestyle='--')
        plt.show()
        for i in range(len(eigenvalues)):
            if eigenvalues[i] < 1:
                break
        index = i-1
        print(index)
        return index
    
    def plt_pca_results(self, index = None, location = '',  plot_components = True, return_comp=0):

        # np.seterr(divide='ignore', invalid='ignore')
        
        """
        Performs PCA with a specified number of components and plots the results.

        The number of components can be specified with the index parameter. If index is not
        specified, the method will use the plt_scree_kaiser method to identify the number
        of components with eigenvalues greater than 1.

        The method can also save the component images to disk using the location parameter.

        Parameters:
            index (int): The number of principal components to use for PCA.
            location (str): The file path to save the component images.
            plot_components (bool): Whether to plot the component images.
            return_comp (int): The component to return as the method result.

        Returns:
            (numpy array): The specified component image as a numpy array.
        """
        if index:
            pca = self.run_pca(index)
            
        else:
            index = self.plt_scree_kaiser() 
            pca = self.run_pca(index)

        self.pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        
  
        if plot_components:

            for i in range(index):

                data = self.pca_scores[:,i]
        
            
      
              
                fig = plt.figure()
                
                da = data.reshape(self.x, self.y)
                masked_array = np.ma.array(da, mask=np.isnan(da))
                masked_array = masked_array/np.max(masked_array)*255
                # masked_array = self._gaussian(masked_array)
                cmap = matplotlib.cm.gist_rainbow
                cmap.set_bad('white',1.)
                plt.imsave(location+'pca'+str(i)+'.tiff',masked_array.astype('uint8'), cmap=cmap)
                plt.close(fig) 
                plt.imshow(masked_array)
                plt.show()


                # fig = make_subplots(rows=1, cols=2, subplot_titles=[ 'PC component '+str(i), 'Pseudo mass spectrum: PC'+str(i)], column_widths=[0.40,0.60])
                # trace1 = show_image_total_ion(self.pca_scores[:,i].reshape(self.x, self.y), title=title, figshow=False)
                # fig.add_trace(trace1.data[0], row=1, col=1)
                
              
                # trace2 = plot_spectrum(self.mz, self.pca_loadings.T[i], title, showfig=False)  
                # fig.add_trace(trace2[0][0], row=1, col=2)
                # fig.add_trace(trace2[0][1], row=1, col=2)
                # fig.update_layout(height = 400, width = 700, coloraxis = dict(colorscale='viridis'))

                # fig.show()

        
        return self.pca_scores[:,return_comp].reshape(self.x, self.y)


    def get_pca_res(self,pc):
        """
        Retrieve the PCA result for a specified principal component.

        This method returns the PCA scores reshaped to the original image dimensions 
        and the corresponding loadings for a specified principal component.

        Parameters
        ----------
        pc : int
            The index of the principal component to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - numpy.ndarray: The PCA scores reshaped to (x, y) dimensions.
            - numpy.ndarray: The loadings for the specified principal component.
        """

        return self.pca_scores[:,pc].reshape(self.x, self.y), self.pca_loadings.T[pc]

    def plt_elbow_method(self):

        """
        Plot the elbow method for KMeans clustering.

        The elbow method is a method of determining the optimal number of clusters
        for KMeans clustering. The method plots the inertia of the model against the
        number of clusters. The point at which the rate of decrease of inertia is
        the highest is considered the optimal number of clusters.

        This method plots the inertia for 1 to 10 clusters and shows the result.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(self.pca_scores)

        inertias = []
        for i in range(1,11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.pca_scores)
            inertias.append(kmeans.inertia_)    
        plt.figure(figsize=(3,3))
        plt.plot(range(1,11), inertias, marker='o')
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

    def run_kmeans(self, n_clusters, dim_red_met = 'PCA', **kwargs):

        """
        Perform KMeans clustering on the data.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to form.
        dim_red_met : str, optional
            The dimensionality reduction method to use. Can be 'PCA' or 'NMF'.
            Default is 'PCA'.
        **kwargs : 
            Additional keyword arguments to show_image_total_ion.

        Returns
        -------
        None

        Notes
        -----
        The function will first perform dimensionality reduction using PCA or NMF, depending on the value of dim_red_met.
        Then, it will perform KMeans clustering on the reduced data and store the labels in the labels attribute of the object.
        Finally, it will display the segmentation result as an image and plot the clusters in the reduced data space.
        """
        if dim_red_met == 'PCA':
            dim_red_df = self.pca_scores
        elif dim_red_met == 'NMF':
            dim_red_df = self.W

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(dim_red_df)
        self.labels = kmeans.labels_

        u_labels = np.unique(self.labels)
        centroids = kmeans.cluster_centers_
      
        # plt.figure(figsize=(7,5))
        # im = plt.imshow(self.labels.reshape(self.x, self.y), interpolation='spline16')


        # # get the colors of the values, according to the 
        # # colormap used by imshow
        # colors = [ im.cmap(im.norm(value)) for value in u_labels]
        # # create a patch (proxy artist) for every color 
        # patches = [ mpatches.Patch(color=colors[i], label="Group {l}".format(l=u_labels[i]) ) for i in range(len(u_labels)) ]
        # # put those patched as legend-handles into the legend
        # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        # plt.show()
        show_image_total_ion(self.labels.reshape(self.x, self.y), title='kmeans', **kwargs)
        self.get_labeled_image(self.labels.reshape(self.x, self.y), u_labels, title = 'Segmentation')
        # if show_PCA:
            # plt.figure(figsize=(5,5))
            # for i in u_labels:
            #     plt.scatter(dim_red_df[self.labels == i, 0] , dim_red_df[self.labels== i, 1] ,label=i, c=colors[i])
            # plt.scatter(centroids[:,0], centroids[:,1], s = 80, color = 'k')
            # plt.xlabel('PC1')
            # plt.ylabel('PC2')
            # plt.legend()
            # plt.show()

            # fig = plt.figure()
            # ax = plt.axes(projection ='3d')
            # for i in u_labels:
            #     ax.scatter(dim_red_df[self.labels == i, 0], dim_red_df[self.labels == i, 1], dim_red_df[self.labels == i, 2],label = i, c=colors[i])
            # plt.show()

        self.bkg_label = self.labels[0] #cause it's in the corner
        print(self.bkg_label, ' -bkg label') 



    def run_bisecting_kmeans(self, nclusters, random_state= 0):

        """
        Perform bisecting k-means clustering on the intensity matrix.

        Parameters:
        nclusters (int): Number of clusters to form.
        random_state (int, optional): Random state for reproducibility. Default is 0.

        This function utilizes the BisectingKMeans algorithm to cluster the 
        flattened intensity matrix (`int_mat_flat`) into the specified number of 
        clusters. After fitting, it stores the labels, results, and cluster centers 
        as attributes (`bm_labels`, `bm_results`, and `bm_clus_cent` respectively).
        Additionally, it visualizes the clustering results and the PCA scores.
        """

        bisect_means = BisectingKMeans(n_clusters = nclusters, random_state=random_state)
        # bisect_means = BisectingKMeans(bisecting_strategy='largest_cluster')
        bisect_means.fit(self.int_mat_flat)
        self.bm_labels = bisect_means.labels_
        self.bm_results = bisect_means.predict(self.int_mat_flat)
        self.bm_clus_cent = bisect_means.cluster_centers_

        show_image_total_ion(self.bm_labels.reshape(self.x,self.y), 'bm_label', figshow=True)
        plt.scatter(self.pca_scores[:,0], self.pca_scores[:,1], c = self.bm_labels)
        plt.scatter(self.bm_clus_cent[:,0], self.bm_clus_cent[:,1], c='b', s=100)
        plt.show()
        print(self.bm_results.shape)

    def extract_roi(self,roi_mask):
        """
        Extract ROI from the intensity matrix.

        Parameters
        ----------
        roi_mask : 2D numpy array
            Binary mask where ROI is 1 and background is 0.

        Returns
        -------
        roi : 2D numpy array
            ROI intensity matrix.
        n_roi : int
            Number of pixels in ROI.

        Notes
        -----
        The function stores the ROI labels and background labels as attributes
        (`roi_labels` and `roibkg_labels`, respectively).
        """
        self.roi_labels = np.where(roi_mask != 0) #take both hemisheperes
        self.roibkg_labels = np.where(roi_mask == 0)
        # print(len(self.roi_labels[0]), ' pixels in total in ROI')
        roi = self.int_mat[self.roi_labels]
        return roi, len(self.roi_labels[0])
    
    def create_masked_roi(self, norm=False):

        
        """
        Create a masked Region of Interest (ROI) from the intensity matrix.

        This method masks the ROI specified by `roi_labels` in the intensity matrix.
        If normalization is enabled, each spectrum in the ROI is normalized by its
        total ion count.

        Parameters
        ----------
        norm : bool, optional
            Whether to normalize the ROI intensity spectra by their total ion count.
            Defaults to False.

        Returns
        -------
        imat
            An instance of the `imat` class representing the masked ROI, with the
            same mass channels as the original intensity matrix.

        Notes
        -----
        - The method modifies the `roi_mat_msk` attribute to store the masked ROI.
        - Normalization is applied to each spectrum individually when `norm` is True.
        """

        tiss_mat = self.int_mat[self.roi_labels]
        roi_mat_msk = np.zeros(self.int_mat.shape)

        if norm :
            print('Normalising...')
            norm_tic = [np.divide(tiss_mat[i], 
                                  np.sum(tiss_mat[i])) for i in range(tiss_mat.shape[0])]

            # norm_tiss_mat = np.divide(tiss_mat,
            #       np.sum(tiss_mat, axis = 0))
            roi_mat_msk[self.roi_labels] = norm_tic
        else: 
            roi_mat_msk[self.roi_labels] = tiss_mat

        self.roi_mat_msk = imat(roi_mat_msk, self.mz)
        # show_image_total_ion(self.roi_mat_msk.tot_ion_int_mat,
                            #   'masked tissue')
        return self.roi_mat_msk
    
    def create_image_avg_roi(self,roi_mean):

        
        """
        Create a masked Region of Interest (ROI) with an average spectrum.

        This method masks the ROI specified by `roi_labels` in the intensity matrix
        and replaces the spectra in the ROI with the average spectrum provided by
        `roi_mean`.

        Parameters
        ----------
        roi_mean : array-like
            The average spectrum to be used to replace the spectra in the ROI.

        Returns
        -------
        imat
            An instance of the `imat` class representing the masked ROI, with the
            same mass channels as the original intensity matrix.

        Notes
        -----
        - The method modifies the `roi_avg_msk` attribute to store the masked ROI.
        """
        roi_mat = np.array([[roi_m]*len(self.roi_labels[0]) for roi_m in roi_mean])
        
        roi_avg_msk = self.int_mat
   

        # if norm :
            # print('Normalising...')
            # norm_tic = [np.divide(tiss_mat[i], 
            #                       np.sum(tiss_mat[i])) for i in range(tiss_mat.shape[0])]

            # # norm_tiss_mat = np.divide(tiss_mat,
            # #       np.sum(tiss_mat, axis = 0))
            # roi_mat_msk[self.roi_labels] = norm_tic
        # else: 
        roi_avg_msk[self.roi_labels] = roi_mat.T

        self.roi_avg_msk = imat(roi_avg_msk, self.mz)
        # show_image_total_ion(self.roi_mat_msk.tot_ion_int_mat,
                            #   'masked tissue')
        return self.roi_avg_msk



    def create_binary_mask(self, labels_roi = None, hdb=False, gmm=False):

        """
        Create a binary mask for the given ROI labels.

        Parameters
        ----------
        labels_roi : array_like, optional
            The ROI labels to be used for creating the binary mask.
            If not provided, the tissue labels will be used.
        hdb : bool, optional
            Whether to use the HDBSCAN labels.
        gmm : bool, optional
            Whether to use the GMM labels.

        Notes
        -----
        The method creates a binary mask where the ROI labels are set to 1 and the
        background labels are set to 0. The binary mask is then reshaped to match
        the original image size and stored as an attribute (`binary_mask_im`).
        The method also updates the `bkg_labels` and `tissue_labels` attributes to
        store the background and tissue labels, respectively.
        """
        binary_mask = np.zeros(self.x*self.y)
        
        if hdb:
            labels = self.hdb_labels
        elif gmm:
            labels = self.gmm_labels
        else:
            labels = self.labels

        if labels_roi:
            roi_labels = np.where(np.isin(labels,labels_roi))
            
            msk_labels = np.where(np.isin(labels,labels_roi) == False)
        else:
            roi_labels = np.where(labels != self.bkg_label)
            msk_labels = np.where(labels == self.bkg_label)

        binary_mask[msk_labels] = 0
        binary_mask[roi_labels] = 1

        self.binary_mask = binary_mask
        self.bkg_labels = msk_labels
        self.tissue_labels = roi_labels
        self.binary_mask_im = binary_mask.reshape(self.x,self.y)
        # plt.figure(figsize=(5,5))
        # plt.imshow(binary_mask.reshape(self.x,self.y))

    def create_masked_tissue(self, umap=False, norm = False):
        #  maybe change the 0 values to np.nan but cannot use it with tsne i think
        #  data_copy[data_copy == 0] = np.nan 
        
        """
        Create a masked intensity matrix from the original intensity matrix by
        selecting the tissue labels and setting the background labels to zero.

        Parameters
        ----------
        umap : bool, optional
            Whether to perform UMAP dimensionality reduction on the intensity
            matrix before masking. Defaults to False.
        norm : bool, optional
            Whether to normalize the intensity matrix by the total ion current
            of each spectrum. Defaults to False.

        Returns
        -------
        None
        """
        tiss_mat = self.int_mat_flat[self.tissue_labels]
        bkg_mat = self.int_mat_flat[self.bkg_labels]
        int_mat_msk = np.zeros(self.int_mat_flat.shape)
        int_mat_bkg = np.zeros(self.int_mat_flat.shape)

        if norm :
            print('Normalising...')
            norm_tiss_mat = np.divide(tiss_mat,
                  np.sum(tiss_mat, axis = 0))
            int_mat_msk[self.tissue_labels] = norm_tiss_mat
        else: 
            int_mat_msk[self.tissue_labels] = tiss_mat
            int_mat_bkg[self.bkg_labels] = bkg_mat

        self.int_mat_msk = imat(self.ravel_matrix(int_mat_msk), self.mz)
        self.int_mat_bkg = imat(self.ravel_matrix(int_mat_bkg), self.mz)
        fig = show_image_total_ion(self.int_mat_msk.tot_ion_int_mat,
                              'masked tissue')
        fig.show()
        

 



    def create_masked_umap(self, n_emb, savefig = True, path=''):
        """
        Create a masked UMAP plot for the given embedding.

        Parameters
        ----------
        n_emb : int
            The embedding number to be used for creating the masked UMAP plot.
        savefig : bool, optional
            Whether to save the figure to a file. Defaults to True.
        path : str, optional
            The file path to save the figure when savefig is True. Defaults to an empty string.

        Returns
        -------
        int_mat_msk : array_like
            The masked UMAP plot as a 2D array.
        """
        tiss_mat = self.embedding[:,n_emb][self.tissue_labels]
        int_mat_msk = np.zeros(self.x*self.y)
        int_mat_msk[self.tissue_labels] = tiss_mat
        int_mat_msk[self.bkg_labels] = [12]*len(self.bkg_labels)

        int_mat_msk = int_mat_msk.reshape(self.x,self.y)
        plt.imshow(int_mat_msk)
        if savefig:
            plt.imsave(path, int_mat_msk)
        return int_mat_msk

    def get_labeled_image(self, data, labels,title):

        """
        Display a labeled image with a legend.

        Parameters
        ----------
        data : array-like
            The data to be reshaped and displayed as an image.
        labels : array-like
            The labels corresponding to the data, used for coloring the image.
        title : str
            The title of the plot.

        Notes
        -----
        The function creates a plot of the data reshaped into the specified dimensions
        with each label assigned a unique color. A legend is added to identify each 
        group by its label. The axis is hidden to focus on the image content.
        """

        u_labels = np.unique(labels)
        # centroids = kmeans.cluster_centers_
        plt.figure(figsize=(7,5))
        im = plt.imshow(data.reshape(self.x, self.y))


        # get the colors of the values, according to the 
        # colormap used by imshow
        colors = [ im.cmap(im.norm(value)) for value in u_labels]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], 
                                   label="Group {l}".format(l=u_labels[i]) ) for i 
                                   in range(len(u_labels)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.title(title)
        plt.axis('off')
        plt.show()


    def run_tsne(self, title = '',learning_rate='auto'):
        
        """
        Run the t-SNE algorithm on the intensity matrix with the background labels set to zero.

        Parameters
        ----------
        title : str, optional
            The title of the plot. Defaults to an empty string.
        learning_rate : str, optional
            The learning rate for the t-SNE algorithm. Defaults to 'auto'.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object of the t-SNE plot.
        data : array_like
            The t-SNE results as a 2D array.
        """
        tsne = TSNE(learning_rate=learning_rate)
        
        tsne_data = self.int_mat_flat.copy()
        tsne_data[self.bkg_labels] = 0
        self.tsne_results = tsne.fit_transform(tsne_data)
        # plt.scatter(self.tsne_results[:,0], self.tsne_results[:,1], s = 10, c = self.labels, alpha = 0.4)
        # plt.show()
        # plt.imshow(self.tsne_results[:,1].reshape(self.x, self.y), cmap='viridis', interpolation='hamming')
        # plt.show()
        
        # marker = self.tsne_results[:,1][100]
        data = self.tsne_results[:,1].copy()
        data[self.bkg_labels] = np.nan
        # data[data == marker] = np.nan
        fig = show_image_total_ion(data.reshape(self.x, self.y), title = 'TSNE '+title)
        fig.update_layout(coloraxis = dict(colorscale='viridis'), plot_bgcolor='rgba(0, 0, 0)', paper_bgcolor='rgba(0,0,0)')
        
        return fig, data
    
    def set_bkg_labels(self, roi_labels = 0):
        
        """
        Set the background labels based on the bkg_label attribute.

        The function searches for the labels equal to bkg_label and stores them
        in the bkg_labels attribute.
        """
        if roi_labels == 0:
            labels = self.labels
            roi_labels = np.where(labels == self.bkg_label)
            # kg_labels = np.where(labels != self.bkg_label)
                
            self.bkg_labels = roi_labels
        else:
            self.bkg_labels = roi_labels
        
       

    def validate_segmentation(self):
        """
        Validate the segmentation using clustering and SHAP values.

        This function uses LGBMClassifier to fit a model to the data and then
        uses SHAP values to compute the importance of each feature. The
        function then plots the SHAP values as a bar chart.

        The function also prints the weighted F1 score for each cluster
        using cross-validation.

        Returns
        -------
        None
        """
        import shap
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_score
                #Setting the objects to category 
        lgbm_data = self.int_mat_flat.copy()
        # for c in lgbm_data.select_dtypes(include='object'):
        #     lgbm_data[c] = lgbm_data[c].astype('category')

        #KMeans clusters
        clf_km = LGBMClassifier(colsample_by_tree=0.8)
        cv_scores_km = cross_val_score(clf_km, lgbm_data, self.labels, scoring='f1_weighted')
        print(f'CV F1 score for K-Means clusters is {np.mean(cv_scores_km)}') 

        #Fit the model
        clf_km.fit(lgbm_data, self.labels)

        #SHAP values
        explainer_km = shap.TreeExplainer(clf_km)
        shap_values_km = explainer_km.shap_values(lgbm_data)
        shap.summary_plot(shap_values_km, lgbm_data, plot_type="bar", plot_size=(15, 10))  

            
    def run_umap(self, umap_metric = 'cosine', title='', 
                        blur=True, savefig= False, location = '', **kwargs):
        # reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, n_components=50, metric='cosine')
        # cosine -meh
        # correlation - good for some but not others
        # euclidean - good at possibly identifying tbi
        # canberra - good tbi segmentation for one smaple only - 2
        # braycurtis
        # mahalanobis takes a loooong time too long
        """
        Run the UMAP algorithm on the intensity matrix with the background labels set to zero.

        Parameters
        ----------
        umap_metric : str, optional
            The metric to use for the UMAP algorithm. Defaults to 'cosine'.
        title : str, optional
            The title of the plots. Defaults to an empty string.
        blur : bool, optional
            If True, the data will be blurred before running UMAP. Defaults to True.
        savefig : bool, optional
            If True, the images will be saved as tiffs. Defaults to False.
        location : str, optional
            The location to save the images. Defaults to an empty string.

        Returns
        -------
        figs : list
            A list of the figures created.
        """
        reducer = umap.UMAP(metric=umap_metric, random_state=42, low_memory=True, **kwargs)
        # reducer = umap.UMAP(random_state=42, low_memory=True)
        # reducer = umap.UMAP()
        # reducer = umap.umap(nneighbors=15,gamma=1.0,nepochs=None,alpha=1.0, 
        #                       spread=1.0,mindist=0.1, 
        #                      a=None,b=None, randomstate=None, verbose=True)
        # #todo: optimise hyperparameters

        # TODO: check mask bug#
        int_mat_flat = self.int_mat_flat.copy()
        data_frame = pd.DataFrame(int_mat_flat)
        data_frame = data_frame.fillna(0)
        # data_frame = median(data_frame, size=3)
       
        data_frame = self._hot_spot_removal(data_frame, 90, impute=False) 
        # 97

        if blur== True:
            data_frame = self.median(data_frame,size=2) 
            # 2
            print(data_frame.shape)
        
            self.embedding = reducer.fit_transform(data_frame)
        else:
            self.embedding = reducer.fit_transform(data_frame)
        
        
        figs = []
        for i in range(self.embedding.shape[1]):
            # plt.imshow(self.embedding[:,i].reshape(self.x, self.y), cmap='viridis',interpolation='hamming')
            # plt.show()
            # plt.imsave('umap'+str(i)+'.tiff',self.embedding[:,i].reshape(self.x, self.y))
            # show_image_total_ion(self.embedding[:,i].reshape(self.x, self.y), title = 'UMAP '+title, figshow=True)
          
            data = self.embedding[:,i]
            if hasattr(self, 'bkg_labels'):
                try:
                    data[self.bkg_labels] = np.nan
                except NameError:
                    pass
            
            if savefig:
                import matplotlib
                fig = plt.figure()
                
                da = data.reshape(self.x, self.y)
                da = exposure.equalize_hist(da)
                masked_array = np.ma.array(da, mask=np.isnan(da))
                masked_array = masked_array/np.max(masked_array)*255
                cmap = matplotlib.cm.viridis
                cmap.set_bad('black',0.)
                plt.imsave(location+'umap'+str(i)+'.tiff',masked_array.astype('uint8'), cmap=cmap)
                plt.close(fig)
                del masked_array
                
            # plt.imshow(data.reshape(self.x, self.y), cmap='viridis',interpolation='hamming')
            # plt.show()
            fig = show_image_total_ion(data.reshape(self.x, self.y), title = 'UMAP '+str(i)+title)
            fig.update_layout(coloraxis = dict(colorscale='viridis'), plot_bgcolor='rgba(0, 0, 0, 0)')
            fig.show()
            figs.append(fig)
            del data
         
            del fig

        return figs


    def run_hdb(self, man_tec = 'TSNE', min_cluster_size = 20):
        """
        Perform clustering on the data using the HDBSCAN algorithm and visualize the results.

        This method applies the HDBSCAN clustering algorithm to the data, either using
        t-SNE or UMAP embeddings, and displays the clustering results.

        Parameters:
        man_tec (str, optional): The manifold learning technique to use for clustering.
                                Options are 'TSNE' for t-SNE and 'UMAP' for UMAP.
                                Defaults to 'TSNE'.
        min_cluster_size (int, optional): The minimum cluster size for HDBSCAN.
                                        Defaults to 20.

        Returns:
        None
        """

        hdb = HDBSCAN(min_cluster_size=min_cluster_size)

        if man_tec == 'TSNE':
            hdb.fit(self.tsne_results)
        elif man_tec == 'UMAP':
            hdb.fit(self.embedding[:,])

            # viewer.add_image(hdb.labels_.reshape(axis[1], axis[2]), colormap='viridis', name='tsne+hdbscan')
        show_image_total_ion(hdb.labels_.reshape(self.x, self.y), title = 'hdbscan', figshow=True)
        self.hdb_labels = hdb.labels_
        print(len(np.unique(hdb.labels_)))
        scatter = plt.scatter(self.embedding[:, 0], self.embedding[:, 1],c=self.hdb_labels, cmap='viridis')
        plt.legend(*scatter.legend_elements())
        #plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the dataset with GMM labels')
        plt.show()
        
    def run_gmm(self, umap = False, n_components = 5, savefig = False, location = ''):
        # doesn't work on the normalised data
        """
        Perform clustering on the data using the Gaussian Mixture Models algorithm and visualize the results.

        Parameters:
        umap (bool, optional): Whether to use UMAP embeddings or the original data for clustering.
                                Defaults to False.
        n_components (int, optional): The number of components to use for the Gaussian Mixture Model.
                                      Defaults to 5.
        savefig (bool, optional): Whether to save the clustering results as an image file.
                                 Defaults to False.
        location (str, optional): The location to save the image file if savefig is True.
                                 Defaults to an empty string.

        Returns:
        None
        """
        
        gmm = GaussianMixture(n_components=n_components)
        if umap == True:
            copy = self.embedding[:,].copy()
            copy[np.isnan(copy)] = 0

            gmm.fit(copy)
            self.gmm_labels = gmm.predict(copy)
          
            # print(self.gmm_labels)
        else:
            gmm.fit(self.int_mat_flat) 
            self.gmm_labels = gmm.predict(self.int_mat_flat)
            # print(self.gmm_labels)

        data = self.gmm_labels
        if hasattr(self, 'bkg_labels'):
            try:
                data[self.bkg_labels] = 0
            except NameError:
                pass

        # plt.scatter(self.pca_scores[:, 0], self.pca_scores[:, 1], 
        #             c=self.gmm_labels, cmap='viridis')

        # plt.title('Gaussian Mixture Models Clustering')
        # plt.show()
        u_labels  =np.unique(self.gmm_labels)
 
        # self.get_labeled_image(self.gmm_labels, self.gmm_labels,'UMAP+GMM' )
        show_image_total_ion(data.reshape(self.x, self.y), title = 'Segmentation', figshow=True)
        if savefig:
            import matplotlib
            fig = plt.figure()
            da = self.gmm_labels.reshape(self.x, self.y).astype('uint8')
            masked_array = np.ma.array(da, mask=np.isnan(da))
            cmap = matplotlib.cm.viridis
            cmap.set_bad('black',0.)
            plt.imsave(location+'gmm.tiff', masked_array, cmap=cmap)
            plt.close(fig)
            
        # plt.title('GMM'+str(n_components))
        plt.show()

        self.get_labeled_image(self.gmm_labels.reshape(self.x, self.y), u_labels, title = 'Segmentation')

        scatter = plt.scatter(copy[:, 0], copy[:, 1],c=self.gmm_labels, cmap='viridis')
        plt.legend(*scatter.legend_elements())
        #plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the dataset')
        plt.show()

    ## from medium:https://medium.com/@quindaly/step-by-step-nmf-example-in-python-9974e38dc9f9
    def rank_calculation(self):
        
        
        # Calculate benchmark value
        """
        Iterate through various values of rank to find optimal rank in NMF. 
        The function will stop when the RMSE of the original df and new V is smaller than the benchmark.
        Benchmark is set as 0.01 of the Frobenius norm of the original df.
        The function will return the best rank and the best V.

        Parameters
        ----------
        self : IMAT object
            The IMAT object containing the data.

        Returns
        -------
        rank : int
            The best rank for NMF.
        V : array-like
            The best V for NMF.
        """
        
        benchmark = np.linalg.norm(df, ord='fro') * 0.0001
        
        # Iterate through various values of rank to find optimal
        rank = 3
        while True:
            
            # initialize the model
            model = NMF(n_components=rank, init='random', random_state=0, max_iter=500)
            W = model.fit_transform(df)
            H = model.components_
            V = W @ H
            
            # Calculate RMSE of original df and new V
            RMSE = np.sqrt(mean_squared_error(df, V))
            
            if RMSE < benchmark:
                return rank, V
            
            # Increment rank if RMSE isn't smaller than the benchmark
            rank += 1
        return rank
    


def show_image_total_ion(tot_ion_int_mat, title, figshow = False, hotspot_removal = False, median_filter = False, savefig = False, location = '', rangecolor=[]):

    # plt.figure(figsize=(5,5))
    # if title:
    #     plt.title(title)
    # else:
    #     plt.title('Total ion image')
    """
    Plot the total ion image.

    Parameters
    ----------
    tot_ion_int_mat : 2D array
        The total ion intensity matrix.
    title : str
        The title of the plot.
    figshow : bool
        If True, show the figure. Defaults to False.
    hotspot_removal : bool
        If True, remove hotspots from the image. Defaults to False.
    median_filter : bool
        If True, apply a median filter to the image. Defaults to False.
    savefig : bool
        If True, save the figure to a file. Defaults to False.
    location : str
        The location to save the figure. Defaults to an empty string.
    rangecolor : list
        The range of colors to use in the heatmap. If empty, use the default
        range of [0, 1].

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object.
    """
    import plotly.express as px
    # remove margins
    # fig = px.imshow(tot_ion_int_mat[5:-5, 5:-5], color_continuous_scale='viridis', aspect='equal')
    if hotspot_removal:
        tot_ion_int_mat = hot_spot_removal(tot_ion_int_mat)
    if median_filter:
        tot_ion_int_mat = median(tot_ion_int_mat, size = 3)
    if len(rangecolor)>0:
        fig = px.imshow(tot_ion_int_mat, color_continuous_scale='viridis', aspect='equal', range_color=rangecolor)
    else:
        fig = px.imshow(tot_ion_int_mat,color_continuous_scale='viridis', aspect='equal')
    fig.update_layout(width = 500, height =400,title = title, 
                      margin=dict(l=0, r=0, b=0, t=40)
                      )
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)
    if figshow:
        fig.show()
    return fig
    # plt.show()
    # if savefig:
    #     plt.imsave(location, tot_ion_int_mat, format = 'tiff', dpi = 1200)

def create_stem_data(x,y, baseline=0.):
    
    """
    Makes y data passing 0 before inbetween actual value to create data for a stem plot
    
    Parameters
    ----------
    x : array
        The x values.
    y : array
        The y values.
    baseline : float
        The value at which to put the baseline.
    
    Returns
    -------
    x : array
        The x values repeated.
    y : array
        The y values repeated and with baseline values inserted.
    """
    x=np.repeat(x,3)
    y=np.repeat(y,3)
    y[::3]=y[2::3]=baseline
    return x,y


def plot_spectrum(x, y, title, showfig = True, color = 'blue'):
        
    # plt.figure(figsize=(14,5))
    # plt.rcParams.update({'font.size': 10})
    # plt.stem(x, y, markerfmt = ' ', linefmt = 'C7-')
    # plt.xlim(np.min(x), np.max(x))
    # plt.ylabel('Ion intensity')
    # plt.xlabel('m/z')
    # # plt.title(title)
    # # plt.ylim(0,0.5)
    # plt.show()
    """
    Plot a mass spectrum using plotly.

    Parameters
    ----------
    x : array_like
        The m/z values for the spectrum.
    y : array_like
        The intensity values for the spectrum.
    title : str
        The title of the plot.
    showfig : bool, optional
        Whether to show the figure. Defaults to True.
    color : str, optional
        The color for the spectrum. Defaults to 'blue'.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure object.
    """
    import plotly.graph_objects as go
    x,y = create_stem_data(x,y)
    traces = []
    traces.append( [ go.Scatter(x=x, y=y, # plot spectra
                        mode='lines', line_color=color, opacity=0.8,
                        showlegend = False, hoverinfo='none'),
                    go.Scatter(x=x, y=y,         
                        mode='markers',marker_color= color, opacity = 0.6,showlegend = False
                        )]  
                        
                     )
    fig = go.Figure()
    fig.add_traces([traces[0][0], traces[0][1]])
    fig.update_layout(title='Mass Spectra: '+title, xaxis_title = 'm/z (Da)', 
                      yaxis_title = 'Intensity',
                    yaxis_zeroline=True, xaxis_zeroline=False, 
                    xaxis_range=[np.min(x),np.max(x)],
                    
                    )

    # fig.update_yaxes(type='log')
    if showfig:
        fig.show()
    return traces


  
        
    # def get_imat(self, index):
        # now redundant
    #     if len(index) == 1:
    #         return self.int_mat[:,:,index]
    #     elif len(index) > 1:
            
    #         return [self.int_mat[:,:,i] for i in index]


class FaissKNeighbors:
    import faiss
    def __init__(self, k=10):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

