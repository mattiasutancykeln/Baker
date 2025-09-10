FROM condaforge/mambaforge:24.3.0-0

# Create non-root user
RUN useradd -m -u 1000 sandboxuser

# Create minimal env with required libs
# rdkit via conda-forge; pandas/numpy/scikit-learn/pyarrow also from conda-forge
RUN mamba install -y -n base -c conda-forge \
    python=3.11 \
    pandas numpy scikit-learn pyarrow \
    rdkit \
    ipykernel \
  && mamba clean -afy

# Create working dirs
RUN mkdir -p /workspace /opt/sandbox && chown -R sandboxuser:sandboxuser /workspace /opt/sandbox
WORKDIR /opt/sandbox

# Copy daemon code
COPY sandbox_daemon.py snapshot_utils.py /opt/sandbox/
RUN chown -R sandboxuser:sandboxuser /opt/sandbox

# Switch to non-root
USER sandboxuser

# Entrypoint: run the daemon (socket lives under /workspace/.sandbox)
ENTRYPOINT ["python", "-u", "/opt/sandbox/sandbox_daemon.py"]


