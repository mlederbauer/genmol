conda create -n genmol python==3.10 -y
conda activate genmol
pip install -r env/requirements.txt
pip install -e .
pip install "setuptools<78"
pip install scikit-learn==1.2.2     # required to run hit generation (gsk3b, jnk3)
conda install -c conda-forge rdkit

# Fix tdc/rdkit compatibility: rdkit.six was removed in modern rdkit versions
python -c "
import site, os
path = os.path.join(site.getsitepackages()[0], 'tdc/chem_utils/oracle/oracle.py')
with open(path) as f: content = f.read()
with open(path, 'w') as f: f.write(content.replace('from rdkit.six import iteritems', 'from six import iteritems'))
"

# Fix rdkit/pandas compatibility: get_adjustment moved to pandas.io.formats.printing in pandas 2.3+
python -c "
import site, os
path = os.path.join(site.getsitepackages()[0], 'rdkit/Chem/PandasPatcher.py')
with open(path) as f: content = f.read()
with open(path, 'w') as f: f.write(content.replace('pandas_formats.format, get_adjustment_name', 'get_adjustment_module, get_adjustment_name'))
"