from .version import __version__, __versiondate__, __license__
from .utils         import *
from .networks      import *
from .interventions import *
from .diseases      import *
from .connectors    import *

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

# Import the version and print the license
print(__license__)

# Double-check key requirements -- should match setup.py
sc.require(['sciris>=3.1.6', 'pandas>=2.0.0', 'scipy', 'numba', 'networkx'], message=f'The following dependencies for STIsim {__version__} were not met: <MISSING>.')
del sc # Don't keep this in the module