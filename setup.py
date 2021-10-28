from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'DLDenoise'
LONG_DESCRIPTION = 'A small library to collect trained low dose CT denoising models and make them easily usable and re-trainable.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="deployable", 
        version=VERSION,
        author="Mayank Patwari",
        author_email="<mpatwari94+work@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: Alpha",
            "Intended Audience :: Siemens CT Concepts",
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
        ]
)