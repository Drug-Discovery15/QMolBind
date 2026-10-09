# NOT built/verified on the authoring machine (Docker daemon was not running) - see DECISIONS D20.
FROM condaforge/miniforge3:latest
WORKDIR /app
# conda-forge chemistry stack first (cacheable layer); the editable pip install needs the sources present
COPY environment.yml pyproject.toml ./
COPY qumolbind ./qumolbind
RUN mamba env create -f environment.yml && mamba clean -afy
ENV PATH=/opt/conda/envs/qumolbind/bin:$PATH OMP_NUM_THREADS=1
COPY . .
# Without an OpenCL GPU the fast oracle uses the (very slow) CPU platform; the unit tests and import check still run.
RUN python -c "import qumolbind, openmm, rdkit, pennylane, qiskit; print('imports ok')"
CMD ["python", "scripts/tasks.py", "test"]
