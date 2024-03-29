======
Theory
======

The derivations given here are treated in more detail in the relevant
literature, e.g. :cite:`Saleh_1991` and :cite:`Goodman_2005`.

Optical transfer function
~~~~~~~~~~~~~~~~~~~~~~~~~

Let us consider a wave field :math:`u(\mathbf{r_0})` whose values we know 
at an initial plane :math:`\mathbf{r_0}=(x_0,y_0,z_0)` (:math:`z_0` fixed).
The field has a certain vacuum wavelength :math:`\lambda` and is traveling through a homogeneous
medium with refractive index :math:`n_\mathrm{m}`. 
From the knowledge of the wave field at the plane :math:`\mathbf{r_0}` and its 
wavelength :math:`\lambda/n_\mathrm{m}`, we can infer the direction of propagation
of the wave field for every point in :math:`\mathbf{r_0}`.
We rewrite the field at :math:`\mathbf{r_0}` as an angular spectrum, a sum over all
possible directions :math:`\mathbf{s}=(p,q,M)`, assuming that the field is only
traveling from left to right

.. math::
	u(\mathbf{r_0}) &= \iint \!\! dp dq \, A(p,q) e^{ik_\mathrm{m}(px_0+qy_0+Mz_0)} \\
	|\mathbf{s}| &= p^2 + q^2 + M^2 = 1 \\
	M &= \sqrt{1-p^2-q^2}. \\

The equation above describes the Huygens-Fresnel principle: the value of the field
:math:`u` at a certain position :math:`\mathbf{r_0}` at the initial plane (point source)
is defined as an integral over all possible plane waves with 
wavenumber :math:`k_\mathrm{m}=\frac{2\pi n_\mathrm{m}}{\lambda}`, 
weighted with the amplitude :math:`A(p,q)`.

Let us now consider the 2D Fourier transform of :math:`u(\mathbf{r_0})`.

.. math::
    \widehat{U}_0(k_\mathrm{x},k_\mathrm{y}) 
    &= \frac{1}{2 \pi} \iint \!\! dx_0 dy_0 \iint \!\! dp dq \, A(p,q)
    e^{ik_\mathrm{m}(px_0+qy_0+Mz_0)} e^{-i(k_\mathrm{x}x_0 +k_\mathrm{y}y_0)} \\
    &= \frac{1}{2 \pi} \iint \!\! dx_0 dy_0 \iint \!\! dp dq \, A(p,q) 
    e^{ik_\mathrm{m}Mz_0} e^{ix_0(k_\mathrm{m}p-k_\mathrm{x})} e^{iy_0(k_\mathrm{m}q-k_\mathrm{y})} \\
    &= \frac{2 \pi}{k_\mathrm{m}^2} A(k_\mathrm{x},k_\mathrm{y}) e^{ik_\mathrm{m}Mz_0}

Here we made use of the identity of the delta distribution

.. math::
    \frac{1}{2 \pi} \int \!\! dx_0 \, e^{ix_0(k_\mathrm{m}p-k_\mathrm{x})}
     = \delta(k_\mathrm{m}p - k_\mathrm{x})
     = \frac{1}{k_\mathrm{m}} \delta(p - k_\mathrm{x}/k_\mathrm{m}) \\
    \frac{1}{2 \pi} \int \!\! dy_0 \, e^{iy_0(k_\mathrm{m}q-k_\mathrm{y})}
     = \delta(k_\mathrm{m}q - k_\mathrm{y})
     = \frac{1}{k_\mathrm{m}} \delta(q - k_\mathrm{y}/k_\mathrm{m})

If we now perform the same procedure for a different position :math:`\mathbf{r_\mathrm{d}}=(x_0,y_0,z_\mathrm{d})`,
we will see that the Fourier transform of the field becomes

.. math::
    \widehat{U}_\mathrm{d}(k_\mathrm{x},k_\mathrm{y}) 
    = \frac{2 \pi}{k_\mathrm{m}^2} A(k_\mathrm{x},k_\mathrm{y}) e^{ik_\mathrm{m}Mz_\mathrm{d}}.

Thus, the propagation of the field :math:`u(\mathbf{r_0})` by a distance :math:`d=z_\mathrm{d}-z_0`
is described by a multiplication with the transfer function 

.. math::
    \mathcal{H}^\text{Helmholtz} &= e^{ik_\mathrm{m}Md} \\

in Fourier space. This is the basis of the convolution-based numerical propagation algorithms implemented in nrefocus.
The process of numerical propagation with the angular spectrum method can be written as

.. math::
	u(\mathbf{r_d}) = \mathcal{F}^{-1}\!\left\lbrace\mathcal{F}\!\left\lbrace u(\mathbf{r_0})\right\rbrace\cdot e^{ik_\mathrm{m}Md}\right\rbrace

with the Fourier transform :math:`\mathcal{F}` and its inverse :math:`\mathcal{F}^{-1}`. With the convolution operator :math:`\ast`,
we may rewrite this equation to

.. math::
	u(\mathbf{r_d}) = u(\mathbf{r_0}) \ast \mathcal{F}^{-1}\!\left\lbrace e^{ik_\mathrm{m}Md} \right\rbrace.
	

Fresnel approximation
~~~~~~~~~~~~~~~~~~~~~
The Fresnel approximation (or paraxial approximation) uses a Taylor expansion to simplify the
exponent of the transfer function :math:`e^{ik_\mathrm{m}Md}`. The exponent can be rewritten as 

.. math::
	ik_\mathrm{m}Md = ik_\mathrm{m}d \left(1-p^2-q^2\right)^{1/2}. 

If the angles of propagation :math:`\theta_\mathrm{x}` and :math:`\theta_\mathrm{y}` for each plane
wave of the angular spectrum is small, then we can make the paraxial approximation:

.. math:: 
	\theta_\mathrm{x} &\approx p \\
	\theta_\mathrm{y} &\approx q \\
	\theta^2 = \theta_\mathrm{x}^2 + \theta_\mathrm{y}^2 &\approx p^2 + q^2
 
We now Taylor-expand the exponent around small values of :math:`\theta`
 
.. math::
    ik_\mathrm{m}d \left(1-\theta^2\right)^{1/2} \approx 
    	ik_\mathrm{m} d\left(1 - \frac{\theta^2}{2} + \frac{\theta^4}{8} - \dots \right). 
 
The Fresnel approximation discards the third term (:math:`\sim \theta^4`) and the transfer function then reads:

.. math::
	e^{ik_\mathrm{m}Md} &\approx e^{ik_\mathrm{m}d} \cdot e^{-\frac{ik_\mathrm{m}d(p^2+q^2)}{2}} \\
	e^{i \sqrt{k_\mathrm{m}^2 - k_\mathrm{x}^2 - k_\mathrm{y}^2 }d} &\approx e^{ik_\mathrm{m}d} \cdot e^{-\frac{id(k_\mathrm{x}^2+k_\mathrm{y}^2)}{2 k_\mathrm{m}}} \\
	\mathcal{H}^\text{Fresnel} &= e^{ik_\mathrm{m}d} \cdot e^{-\frac{id(k_\mathrm{x}^2+k_\mathrm{y}^2)}{2 k_\mathrm{m}}}


Thus, the propagation by a distance distance :math:`d=z_\mathrm{d}-d` in the Fresnel approximation
can be written in the form of the convolution

.. math::
	u(\mathbf{r_d}) = e^{ik_\mathrm{m}d} \cdot u(\mathbf{r_0}) \ast \mathcal{F}^{-1}\!\left\lbrace e^{-\frac{id(k_\mathrm{x}^2+k_\mathrm{y}^2)}{2 k_\mathrm{m}}} \right\rbrace.

Note that the Fresnel approximation results in paraboloidal waves :math:`(p^2+q^2)` whereas spherical
waves are used with the Helmholtz equation.


Transfer functions in nrefocus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The numerical focusing algorithms in this package require the input data :math:`u_\text{in}` to be normalized
by the incident plane wave :matH:`u_0(\mathbf{r_0})` according to 

.. math::
    u_\text{in}(\mathbf{r_0}) = \frac{u(\mathbf{r_0})}{u_0(\mathbf{r_0})}

As a result, the transfer functions change to

.. math::
    \mathcal{H}_\text{norm}^\text{Helmholtz} &= e^{ik_\mathrm{m}(M-1)d} = e^{id\left(\sqrt{k_\mathrm{m}^2 - k_\mathrm{x}^2 - k_\mathrm{y}^2} - k_\mathrm{m}\right)}\\
    \mathcal{H}_\text{norm}^\text{Fresnel} &= e^{-\frac{id(k_\mathrm{x}^2+k_\mathrm{y}^2)}{2 k_\mathrm{m}}}.
