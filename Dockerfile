FROM condaforge/miniforge3:latest
WORKDIR /app
COPY environment.yml pyproject.toml ./
COPY qumolbind ./qumolbind
RUN mamba env create -f environment.yml && mamba clean -afy
ENV PATH=/opt/conda/envs/qumolbind/bin:$PATH OMP_NUM_THREADS=1
COPY . .
CMD ["python", "scripts/tasks.py", "smoke"]
