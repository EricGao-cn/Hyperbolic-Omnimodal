from .pmrl import PMRL
from .hyperbolic_spectral_pmrl import HyperbolicSpectralPMRL
from .lorentz_omnimodal import LorentzOmnimodal

model_registry = {
                    'pmrl':PMRL,
                    'hyperbolic_spectral_pmrl':HyperbolicSpectralPMRL,
                    'lorentz_omnimodal':LorentzOmnimodal
                    }
