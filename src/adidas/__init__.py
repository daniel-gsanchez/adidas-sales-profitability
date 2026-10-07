"""Adidas US sales: data audit, market segmentation and operating-margin model."""
import os

# Windows housekeeping, set before any model is fitted:
# - joblib cannot count physical cores on recent Windows versions and warns;
#   it skips the check when given a count below the number of logical cores.
# - scikit-learn's K-means warns about a memory leak with MKL and several
#   OpenMP threads; one thread is plenty for a few hundred markets.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
