FROM jupyter/scipy-notebook

# Install skll: https://skll.readthedocs.org/en/latest/getting_started.html
RUN conda install --yes -c dan_blanchard skll
RUN pip install bigml
