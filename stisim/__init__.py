from .version import __version__, __versiondate__, __license__

from .connectors    import *
from .diseases      import *
from .interventions import *
from .networks      import *
from .plotting      import *
from .products      import *
from .utils         import *

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent
data = root/'data'

# Import the version and print the license
print(__license__)

# Double-check key requirements -- should match setup.py
sc.require(['sciris>=3.1.6', 'pandas>=2.0.0', 'scipy', 'numba', 'networkx'], message=f'The following dependencies for STIsim {__version__} were not met: <MISSING>.')
del sc # Don't keep this in the module