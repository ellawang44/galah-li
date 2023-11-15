import numpy as np
import math
from scipy.interpolate import CubicSpline, interp1d

class Stats:
    def __init__(self, sample=None, x=None, y=None, num=1000):
        if sample is not None: # sampled rvs
            self.xmin = np.min(sample)
            self.xmax = np.max(sample)
            self.x_interp = np.linspace(self.xmin, self.xmax, num)
            self.calc_cdf(sample)
            self.calc_hist(sample)
            self.smooth_hist(N=3)
        else: # analytical rvs
            self.x, self.y = x, y
            self.xmin = x[0]
            self.xmax = x[-1]
            self.cdf_func = CubicSpline(self.x, np.cumsum(y)/np.sum(y))
            self.x_interp = np.linspace(self.xmin, self.xmax, num)
            self.y_interp = CubicSpline(self.x, self.y)(self.x_interp)
        self.calc_MLE()
        self.calc_err()

    def calc_cdf(self, sample):
        '''Emperical cdf, with linear interpolation. 

        Parameters
        ----------
        sample : 1darray
            The sample to calculate the cdf for. 
        '''

        self.cdf_func = interp1d(np.sort(sample), np.linspace(0, 1, len(sample)), fill_value='extrapolate')

    def calc_hist(self, sample):
        '''Calculate the histogram.

        Parameters
        ----------
        sample : 1darray
            The sample to calculate the histogram for. 
        '''

        def ptp(x):
            '''Point to point distance'''
            return np.max(x) - np.min(x)
        def stone(x):
            '''Stone algorithm for finding optimal bin size of a data set. Based on leave-one-out cross-validation estimate of the integrated squared error. Can be regarded as a generalization of Scott’s rule.
            '''
            n = x.size
            _range = (np.min(x), np.max(x))
            ptp_x = ptp(x)
            if n <= 1 or ptp_x == 0:
                return 0

            def jhat(nbins):
                hh = ptp_x / nbins
                p_k = np.histogram(x, bins=nbins, range=_range)[0] / n
                return (2 - (n + 1) * p_k.dot(p_k)) / hh

            nbins_upper_bound = max(100, int(np.sqrt(n)))
            nbins = min(range(1, nbins_upper_bound + 1), key=jhat)
            self.stone_good = True
            if nbins == nbins_upper_bound:
                self.stone_good = False
            return ptp_x / nbins

        width = stone(sample)
        bins = math.ceil((self.xmax-self.xmin)/width)
        hist, edges = np.histogram(sample, bins=bins, range=(self.xmin,self.xmax))
        self.width = (self.xmax-self.xmin)/bins
        self.x = np.mean([edges[:-1], edges[1:]], axis=0)
        self.y = hist/(len(sample)*self.width)
        
    def smooth_hist(self, N=3):
        '''Smooth the histogram using running mean.

        Parameters
        ----------
        N : int, optional
            The number of points in the running mean window. 
        '''

        def running_mean(x, N):
            '''Calculate running mean with window N'''
            return np.convolve(x, np.ones(N)/N, mode='same')

        self.y_interp = interp1d(self.x, running_mean(self.y, N), fill_value='extrapolate')(self.x_interp)

    def calc_MLE(self):
        '''Maximum likelihood. 
        '''

        self.MLE_ind = np.argmax(self.y_interp)
        self.MLE = self.x_interp[self.MLE_ind]
    
    def cross(self, x, y, line):
        '''Find where line intersects the numerical function x, y.

        Parameters
        ----------
        x : 1darray
            The x array to the numerical function.
        y : 1darray
            The y array to the numerical function.
        line : float
            The horizontal line that intersects the numerical function.

        Returns
        -------
        xs : 1darray
            The x values where the line intersects the function, linear interpolation is used.  
        '''

        # find places where y crosses the line
        xs = []
        for ind, (left, right) in enumerate(zip(y[:-1], y[1:])):
            if np.min([left, right]) <= line <= np.max([left, right]):
                grad = (right-left)/(x[ind+1]-x[ind])
                inter = left - x[ind]*grad
                x_target = (line-inter)/grad
                xs.append(x_target)
        xs = np.array(xs)
        return xs

    def calc_err(self, threshold=0.005, num=1000):
        '''Calculate the errors. Uses KDE, unless single sided error.

        Parameters
        ----------
        threshold : float, optional
            The tolerance till convergence. Default 0.5% either side of 68%.  
        '''

        def calc_areas(line):
            '''Calculate all of the areas (pairwise intersects).
            '''

            xs = self.cross(self.x_interp, self.y_interp, line)
            lower_xs = xs[xs < self.MLE]
            upper_xs = xs[self.MLE < xs]
            areas = []
            for l in lower_xs:
                for u in upper_xs:
                    area = self.cdf_func(u) - self.cdf_func(l)
                    areas.append([area, l, u])
            return areas
        
        def closest_area(areas):
            '''Give back the area closest to 68%.
            '''

            diffs = [np.abs(area[0] - 0.68) for area in areas]
            return areas[np.argmin(diffs)]
        
        # starting guess
        line = np.max(self.y_interp)*(1-1/num)
        areas = calc_areas(line)
        best_area = closest_area(areas)
        current_area = best_area
        half = False

        # iterate till optimal line palcement
        while np.abs(current_area[0] - 0.68) > threshold:
            # find all areas
            areas = calc_areas(line)
            # one of the sides has fallen off
            if len(areas) == 0: 
                half = True
                break
            # find the closest area in this batch
            current_area = closest_area(areas)
            # update best area if this is area is better
            if abs(current_area[0] - 0.68) < abs(best_area[0] - 0.68):
                best_area = current_area
            # walked past optimal
            if all([a[0] > 0.68+threshold for a in areas]): 
                break
            line -= np.max(self.y_interp)/num 
        self.line = line

        # calculate x values from line
        if not half:
            self.err_low = best_area[1]
            self.err_upp = best_area[2]
            self.area = best_area[0]
        # reporting 34% error
        if half:
            if (self.y_interp[0] - self.y_interp[-1]) > 0.5*np.max(self.y_interp):
                self.calc_half(side='upper')
            elif (self.y_interp[-1] - self.y_interp[0]) > 0.5*np.max(self.y_interp):
                self.calc_half(side='lower')
            else:
                self.err_low = np.nan
                self.err_upp = np.nan
                self.area = np.nan

    def calc_half(self, side):
        '''Calculate the lower or upper error as 34% away from the MLE. Set other side to np.nan.

        Parameters
        ----------
        side : str
            The side to calculate the error on. Can be lower or upper.
        '''
        
        MLE_cdf = self.cdf_func(self.x_interp)[self.MLE_ind]
        if side == 'lower':
            self.err_upp = np.nan
            if MLE_cdf <= 0.34: # bad distribution
                self.err_low = np.nan
                self.area = np.nan
            else:
                self.err_low = self.cross(self.x, self.cdf_func(self.x), MLE_cdf - 0.34)[0]
                self.area = self.cdf_func(self.MLE) - self.cdf_func(self.err_low)
        elif side == 'upper':
            self.err_low = np.nan
            if MLE_cdf >= (1-0.34): # bad distribution
                self.err_upp = np.nan
                self.area = np.nan
            else:
                self.err_upp = self.cross(self.x, self.cdf_func(self.x), MLE_cdf + 0.34)[0]
                self.area = self.cdf_func(self.err_upp) - self.cdf_func(self.MLE)

