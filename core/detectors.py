import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class _Base(object):
    name = "base"

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.seed = cfg["seed"] if seed is None else seed
        self.scaler = None
        self.keep = None
        self.train_scores = None

    def _prep_fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        v = X.std(axis=0)
        self.keep = v > 1e-12
        if not self.keep.any():
            self.keep = np.ones(X.shape[1], dtype=bool)
        X = X[:, self.keep]
        self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X)

    def _prep(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X[:, self.keep])

    def fit(self, X_normal):
        Z = self._prep_fit(X_normal)
        self._fit(Z)
        self.train_scores = self._score(Z)
        return self

    def score(self, X):
        return self._score(self._prep(X))

    def threshold(self):
        p = self.cfg["detectors"]["threshold_percentile"]
        return float(np.percentile(self.train_scores, p))


def capped_pca(Z, variance, cfg, seed):
    p = PCA(n_components=variance, svd_solver="full", random_state=seed).fit(Z)
    frac = cfg["detectors"].get("n_components_max_frac")
    if not frac:
        return p, None
    cap = max(1, int(np.floor(Z.shape[0] * float(frac))))
    if p.n_components_ <= cap:
        return p, cap
    return PCA(n_components=cap, svd_solver="full", random_state=seed).fit(Z), cap


class MahalanobisDetector(_Base):
    name = "mahalanobis"

    def _fit(self, Z):
        self.pca, self.n_components_cap = capped_pca(
            Z, self.cfg["detectors"]["pca_variance"], self.cfg, self.seed)
        Y = self.pca.transform(Z)
        self.mu = Y.mean(axis=0)
        c = np.cov(Y, rowvar=False)
        c = np.atleast_2d(c)
        self.inv = np.linalg.pinv(c + 1e-12 * np.eye(c.shape[0]))

    def _score(self, Z):
        Y = self.pca.transform(Z) - self.mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Y, self.inv, Y), 0.0))


class PCAReconstructionDetector(_Base):
    name = "pca_recon"

    def _fit(self, Z):
        self.pca, self.n_components_cap = capped_pca(
            Z, self.cfg["detectors"]["pca_reconstruction"]["variance"], self.cfg, self.seed)

    def _score(self, Z):
        R = self.pca.inverse_transform(self.pca.transform(Z))
        return np.sqrt(np.sum((Z - R) ** 2, axis=1))


class IsolationForestDetector(_Base):
    name = "isolation_forest"

    def _fit(self, Z):
        p = self.cfg["detectors"]["isolation_forest"]
        self.m = IsolationForest(n_estimators=p["n_estimators"],
                                 max_samples=p["max_samples"],
                                 random_state=self.seed).fit(Z)

    def _score(self, Z):
        return -self.m.score_samples(Z)


class OneClassSVMDetector(_Base):
    name = "ocsvm"

    def _fit(self, Z):
        p = self.cfg["detectors"]["one_class_svm"]
        self.m = OneClassSVM(nu=p["nu"], gamma=p["gamma"], kernel="rbf").fit(Z)

    def _score(self, Z):
        return -self.m.decision_function(Z)


ALL = [MahalanobisDetector, IsolationForestDetector, OneClassSVMDetector,
       PCAReconstructionDetector]
BY_NAME = {c.name: c for c in ALL}
