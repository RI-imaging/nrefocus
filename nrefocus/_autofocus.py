#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from . import metrics

__all__ = [
            "autofocus_field",
            "autofocus_sinogram",
            "minimize_metric",
          ]

def autofocus_field(Field, nm, res, ival=None, roi=None,
                    norm="average gradient", d=None):
    """ Perform numerical autofocusing
    
    Parameters
    ----------
    Field : 1d or 2d ndarray
        Electric field is BG-Corrected, i.e. Field = EX/BEx
    nm : float
        Refractive index of medeium.
    res : float
        Size of wavelength in pixels.
    ival : tuple of floats
        Approximate interval to search for optimal focus in px.
    roi : rectangular region of interest (x1,y1,x2,y2)
        Region of interest of `Field` for which the norm will be
        minimized. If not given, the entire `Field` will be used.
    d : None or float
        No autofocusing. Refocus to focus position `d`.
    norm : str
        - "average gradient" : average gradient norm of amplitude
        - "rms contrast" : RMS contrast of phase data
        - "spectrum" : sum of filtered Fourier coefficients
        
    Returns
    -------
    The focused field (and the refocussing distance + data if d is None)
    """
    Fshape = len(Field.shape)
    if Fshape == 1:
        propfunc = free_space_propagate_1d
    elif Fshape == 2:
        propfunc = free_space_propagate_2d
    else:
        raise ValueError("Unsupported dimension: {}".format(Fshape))


    if d is not None:
        if roi is None:
            if Fshape == 2:
                roi = (0,0,Field.shape[0], Field.shape[1])
            else:
                roi = (0,Field.shape[0])
        fftfield = np.fft.fftn(Field)   
        #if not Field.dtype == np.dtype(np.complex):
        #    Field = np.array(Field, dtype=np.complex)
        # 
        ## Set up fast fourier transform
        #fftplan = fftw3.Plan(Field.copy(), None, nthreads = _ncores,
        #                     direction="forward", flags=_fftwflags)
        #fftfield = np.zeros(Field.shape, dtype=np.complex)
        #fftplan.guru_execute_dft(Field, fftfield)
        #fftw.destroy_plan(fftplan)

        return propfunc(fftfield, d, nm, res)
    else:
        if norm == "average gradient":
            norm = lambda x: metrics.average_gradient(np.abs(x))
        elif norm == "rms contrast":
            norm = lambda x: -metrics.contrast_rms(np.angle(x))
        elif norm == "spectrum":
            norm = lambda x: metrics.spectral(np.abs(x),res)
        else:
            raise ValueError("No such norm: {}".format(norm))
        
        return minimize_metric(Field, norm, ival, nm, res, roi=roi,
                                 return_gradient=True)


def autofocus_sinogram(sino, nm, res, ival=(None,None),
                    norm="average gradient", d=None, ret_dopt=True,
                    same_dist=False):
    """ Perform numerical autofocusing
    
    Parameters
    ----------
    sino : 2d or 3d ndarray
        Electric field is BG-Corrected, i.e. Field = EX/BEx
    nm : float
        Refractive index of medeium.
    res : float
        Size of wavelength in pixels.
    ival : tuple of floats
        Approximate interval to search for optimal focus in px.
    d : None or float
        No autofocusing. Refocus to focus position `d`.
    norm : str
        see `autofocus_field`.
    ret_dopt : bool
        Return optimized distance and gradient plotting data.
    same_dist : bool
        Refocus entire sinogram with one distance.
    
    Returns
    -------
    The focused field (and the refocussing distance + data if d is None)
    """
    dopt = list()
    grad = list()
    newsino = np.zeros(sino.shape, dtype=np.complex)

    for s in range(len(sino)):
        if d is not None:
            field = autofocus_field(sino[s], nm, res, ival, norm, d=d)
        else:
            field, do, gr = autofocus_field(sino[s], nm=nm, res=res,
                                            ival=ival, norm=norm)
            dopt.append(do)
            grad.append(gr)
        newsino[s] = field

    if same_dist and d is None:
        # find average dopt
        davg = np.average(dopt)
        dopt = [davg]*len(dopt)
        for s in range(len(sino)):
            newsino[s] = autofocus_field(sino[s], nm, res, ival, norm, d=davg)

    if len(dopt) == 0:
        return newsino
    else:
        if ret_dopt:
            return newsino, dopt, grad
        else:
            return newsino



def minimize_metric(field, norm, ival, nm, lambd, roi=None,
                       coarse_acc=1, fine_acc=.005,
                       return_gradient=False):
    """ Find the focus by minimizing the `norm` of an image
    
    Parameters
    ----------
    field : 2d array
        electric field
    norm : callable
        some norm to be minimized
    ival : tuple of floats
        (minimum, maximum) of interval to search in pixels
    nm : float
        RI of medium
    lambd : float
        wavelength in pixels
    roi : rectangular region of interest (x1,y1,x2,y2)
        Region of interest of `field` for which the norm will be
        minimized. If not given, the entire `field` will be used.
    coarse_acc : float
        accuracy for determination of global minimum in pixels
    fine_acc : float
        accuracy for fine localization percentage of gradient change
    return_gradient:
        return x and y values of computed gradient
    """
    if roi is not None:
        assert len(roi) == len(field.shape)*2, "ROI must match field dimension"
    
    Fshape = len(field.shape)
    if Fshape == 1:
        propfunc = free_space_propagate_1d
    elif Fshape == 2:
        propfunc = free_space_propagate_2d
    else:
        raise ValueError("Unsupported dimension: {}".format(Fshape))
    
    if roi is None:
        if Fshape == 2:
            roi = (0,0,field.shape[0], field.shape[1])
        else:
            roi = (0,field.shape[0])
    
    if ival[0] > ival[1]:
        ival = (ival[1], ival[0])
    # set coarse interval
    #coarse_acc = int(np.ceil(ival[1]-ival[0]))/100
    N = 100/coarse_acc
    zc = np.linspace(ival[0], ival[1], N, endpoint=True)
    
    # compute fft of field
    fftfield = np.fft.fftn(field)

    #fftplan = fftw3.Plan(fftfield.copy(), None, nthreads = _ncores,
    #                     direction="backward", flags=_fftwflags)
    
    # initiate gradient vector
    gradc = np.zeros(zc.shape)
    for i in range(len(zc)):
        d = zc[i]
        #fsp = propfunc(fftfield, d, nm, lambd, fftplan=fftplan)
        fsp = propfunc(fftfield, d, nm, lambd)
        if Fshape == 2:
            gradc[i] = norm(fsp[roi[0]:roi[2], roi[1]:roi[3]])
        else:
            gradc[i] = norm(fsp[roi[0]:roi[1]])
    
    minid = np.argmin(gradc)
    if minid == 0:
        zc -= zc[1]-zc[0]
        minid += 1
    if minid == len(zc)-1:
        zc += zc[1]-zc[0]
        minid -= 1
    ival=(zc[minid-1], zc[minid+1])

    numfine = 10
    mingrad = gradc[minid]


    while True:

        gradf = np.zeros(numfine)
        zf = np.linspace(ival[0],ival[1],numfine)
        for i in range(len(zf)):
            d = zf[i]
            #fsp = propfunc(fftfield, d, nm, lambd, fftplan=fftplan)
            fsp = propfunc(fftfield, d, nm, lambd)
            if Fshape == 2:
                gradf[i] = norm(fsp[roi[0]:roi[2], roi[1]:roi[3]])
            else:
                gradf[i] = norm(fsp[roi[0]:roi[1]])
        minid = np.argmin(gradf)
        if minid == 0:
            zf -= zf[1]-zf[0]
            minid += 1
        if minid == len(zf)-1:
            zf += zf[1]-zf[0]
            minid -= 1
        ival=(zf[minid-1], zf[minid+1])
        if abs(mingrad-gradf[minid])/100 < fine_acc:
            break

    minid = np.argmin(gradf)

    if return_gradient:
        return fsp, zf[minid], [(zc, gradc), (zf,gradf)]
    return fsp, zf[minid] 
