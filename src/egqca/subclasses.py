from .constants import angstrom_to_meter
import numpy as np

class lattice:
    """ Store lattice parameters """

    def __init__(self, a, b, c, alpha, beta, gamma):
        """
        Store the lattice parameters in SI units and derive the cell volume.

        Parameters
        ----------
        a, b, c : float
            Lattice parameters in Angstrom.
        alpha, beta, gamma : float
            Lattice angles in degree.
        """

        # Convert to meters and radians
        a_m = a * angstrom_to_meter
        b_m = b * angstrom_to_meter
        c_m = c * angstrom_to_meter

        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
        
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)

        # General triclinic cell volume
        volume_factor = np.sqrt(
            1.0
          + 2.0 * cos_alpha * cos_beta * cos_gamma
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
        )

        # save 
        self.a = a_m
        self.b = b_m
        self.c = c_m
        self.V_cell = a_m * b_m * c_m * volume_factor  # m^3 per cell
