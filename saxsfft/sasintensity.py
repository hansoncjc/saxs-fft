import numpy as np
from abc import ABC, abstractmethod
from .structurefactor import StructureFactor
from .formfactor import Sphere
import matplotlib.pyplot as plt

class Intensity(ABC):
    def __init__(self, volume_fraction, sld_sample, sld_solvent):
        self.volume_fraction = volume_fraction
        self.sld_sample = sld_sample
        self.sld_solvent = sld_solvent
        self.prefactor = self._compute_prefactor()
        self.structure_factor = None
        self.form_factor = None

    def _compute_prefactor(self):
        self.delta_rho = self.sld_sample - self.sld_solvent
        return self.volume_fraction * self.delta_rho**2

    def set_structure_factor(self, gsd_path, N_grid, frames='last:5', device=None, dtype=None):
        self.structure_factor = StructureFactor(gsd_path, N_grid, frames, device=device, dtype=dtype)

    @abstractmethod
    def set_form_factor(self, *args, **kwargs):
        """Set self.form_factor in subclass."""
        pass

    @abstractmethod
    def compute_Iq(self):
        """Compute the intensity I(q) based on structure and form factors."""
        pass
    
    '''
    def plot_Iq(self, q_unit = "angstrom"):
        """Plot the computed intensity I(q) with appropriate labels."""
        
        if not hasattr(self, 'Iq'):
            self.compute_Iq()
        
        if q_unit == "angstrom":
            q = self.qr / (self.radius * 10)
            u_label = '$q(\AA^{-1})$'
        elif q_unit == "nm":
            q = self.qr / self.radius
            u_label = '$q(nm^{-1}$)'
        else:
            raise ValueError("Unsupported q_unit. Use 'angstrom' or 'nm'.")
        
        plt.figure(figsize=(8, 6), dpi=100)
        plt.loglog(q[3:-1], self.Iq[3:-1], label=f'Intensity I(q)')
        plt.xlabel(u_label, fontsize=14)
        plt.ylabel('Intensity(a.u.)', fontsize=14)
    '''



class SphereIntensity(Intensity):
    def set_form_factor(self, radius):
        """Set the form factor for a sphere. radius is in nm."""
        self.radius = radius
        self.form_factor = Sphere(radius)

    def compute_Iq(self):
        """Compute the intensity I(q) for spherical particles."""
        if self.structure_factor is None or self.form_factor is None:
            raise ValueError("Both structure and form factors must be initialized before computing I(q).")
        q, Sq = self.structure_factor.compute_s_1d()
        Pq = self.form_factor.Compute_Pq(q)
        V = 4/3 * np.pi * self.radius**3
        Iq = self.prefactor * V * Sq * Pq
        self.Iq = Iq
        self.q = q
        return q, Iq