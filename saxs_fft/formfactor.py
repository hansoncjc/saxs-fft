import numpy as np
from abc import ABC, abstractmethod

class FormFactor(ABC):
    """Abstract base class for particle form factors."""

    def __init__(self, shape_name):
        self.shape = shape_name  # Fixed typo

    @abstractmethod
    def whatshape(self):
        """Returns the shape name."""
        return self.shape

    @abstractmethod
    def Compute_Fq(self, q):
        """Returns the scattering amplitude F(q) for given q."""
        pass

    def Compute_Pq(self, q):
        """Returns form factor P(q) = |F(q)|^2 for given qr."""
        return np.abs(self.Compute_Fq(q))**2

class Sphere(FormFactor):
    """Class for spherical form factor."""

    def __init__(self, radius):
        super().__init__('sphere')
        self.radius = radius

    def whatshape(self):
        print(self.shape)
        return self.shape

    def Compute_Fq(self, q):
        """
        Scattering amplitude F(q) of spheres.

        Parameters
        ----------
        q : (Nq,) array_like
            Magnitude of scattering vector in *simulation inverse-length units*.
        r : float or (Nr,) array_like
            Sphere radius/radii in *simulation length units, default is 1*.

        Returns
        -------
        Fq : ndarray
            If r is scalar → shape (Nq,);        F(q) for that one radius.
            If r has length >1 → shape (Nq, Nr); F(q) for every q–radius pair.
            The second index corresponds to the order of radii in `r`.
        """
        q  = np.atleast_1d(q).astype(float)
        r  = np.atleast_1d(self.radius).astype(float)

        # Outer product gives every qr combination:  shape (Nq, Nr)
        qr = np.outer(q, r)

        # Compute 3[sin(qr) − qr cos(qr)] / (qr)^3,  with F(0) = 1
        with np.errstate(divide='ignore', invalid='ignore'):
            Fq = 3 * (np.sin(qr) - qr * np.cos(qr)) / qr**3
            Fq = np.where(qr == 0, 1.0, Fq)

        # If only one radius was supplied, squeeze to 1-D for convenience
        if r.size == 1:
            return Fq.squeeze()      # shape (Nq,)
        return Fq                    # shape (Nq, Nr)
