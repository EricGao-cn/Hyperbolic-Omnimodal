from .pmrl import PMRL
from .hyperbolic_spectral_pmrl import HyperbolicSpectralPMRL
from .lorentz_omnimodal import LorentzOmnimodal
from .subset_hierarchy import SubsetHierarchy

model_registry = {
                    'subset_hierarchy':SubsetHierarchy,
                    'pmrl':PMRL,
                    'hyperbolic_spectral_pmrl':HyperbolicSpectralPMRL,
                    'lorentz_omnimodal':LorentzOmnimodal
                    }
