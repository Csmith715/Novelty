from setuptools import setup, find_packages

requirements=[
   'pandas',
   'statsmodels',
   'seaborn',
   'scipy',
   'numpy',
   'scikit-learn',
   'psycopg2==2.7.6.1',
   'sqlalchemy',
   'requests',
   ]

setup(name='precedence',
    version='0.0.1',
    description='Precendence score',
    url='https://gitlab.com/AIPSDev/research/precedence.git',
    author='chris.mirabzadeh.aon',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: AON",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    zip_safe=False)